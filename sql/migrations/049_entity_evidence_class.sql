-- Entities carry an evidence class, so a guess is distinguishable from a fact (TBU-261).
--
-- Today every row in `entities` looks alike. `entity:PostgreSQL`, extracted
-- because it has interior capitalisation, and `location:Paris`, extracted
-- because a gazetteer contains the word, are stored identically and consumed
-- identically. Downstream has no way to tell a syntactic certainty from a
-- dictionary guess, and the moment an enrichment layer (TBU-260) or an adapter
-- (TBU-268) starts writing entities, machine-suggested rows will be
-- indistinguishable from adapter-derived ground truth.
--
-- That is not a hypothetical. It is what `person:` did: a value that looked
-- like signal, multiplied into the entity-overlap ranking term by code written
-- by someone who was not in the conversation where it was defined.
--
-- DELIBERATELY NOT A CONFIDENCE SCORE. Guo, Pleiss, Sun and Weinberger,
-- "On Calibration of Modern Neural Networks" (ICML 2017), establish that
-- calibration is a property of an ESTIMATED probability. What we have here is
-- a fixed constant per rule class -- nothing is estimated, so nothing can be
-- calibrated, and a float column would invite every consumer to multiply it
-- into a score as though it were a probability. It is an enumerated class.
--
-- The mapping is v0.18.0's surviving rule written down: the four types that
-- kept their entity type after `person:` was deleted are exactly the four with
-- an unambiguous syntactic marker.
--
--     syntactic   entity:  CamelCase              file:   path separator
--                 error:   Error/Exception suffix quantity: number + unit
--     inferred    event: activity: emotion: relationship: preference: location:
--                 -- dictionary or keyword lookup, no marker
--     structured  nothing yet -- TBU-268 fills this from adapter provenance
--
-- NOT DONE HERE, deliberately: migration 048 excludes person entities from the
-- spreading-activation walk with `e2.entity_type <> 'person'`. Rewriting that
-- as `e2.evidence_class <> 'inferred'` names the mechanism instead of the
-- symptom and is the eventual goal, but it WIDENS the exclusion -- six types
-- that traverse today would stop. That is a retrieval behaviour change and it
-- needs a LongMemEval run (R@10 0.9972 must not regress) before it ships. 048's
-- own supporting benchmark was retracted for being unsound; this one gets
-- measured first.
--
-- Derived from the tag prefix inside `link_memory_entities`, so no Python and
-- no Go client change is required and TBU-269 parity is not exposed.

-- ── The mapping, in exactly one place ─────────────────────────────────

CREATE OR REPLACE FUNCTION entity_evidence_class(p_entity_type text)
RETURNS text
LANGUAGE sql IMMUTABLE
SET search_path = public, extensions
AS $$
    SELECT CASE p_entity_type
        WHEN 'entity'   THEN 'syntactic'
        WHEN 'file'     THEN 'syntactic'
        WHEN 'error'    THEN 'syntactic'
        WHEN 'quantity' THEN 'syntactic'
        ELSE 'inferred'
    END;
$$;

COMMENT ON FUNCTION entity_evidence_class(text) IS
    'Maps an entity_type to its evidence class (TBU-261). Single source of '
    'truth -- link_memory_entities and the 049 backfill both call this.';

-- Rank for "strongest evidence wins" on upsert. Evidence is monotone: an
-- entity that has ever been seen with adapter provenance keeps that standing,
-- so a later syntactic sighting must not demote it.
CREATE OR REPLACE FUNCTION entity_evidence_rank(p_class text)
RETURNS integer
LANGUAGE sql IMMUTABLE
SET search_path = public, extensions
AS $$
    SELECT CASE p_class
        WHEN 'structured' THEN 3
        WHEN 'syntactic'  THEN 2
        ELSE 1
    END;
$$;

-- ── Column ────────────────────────────────────────────────────────────

ALTER TABLE entities
    ADD COLUMN IF NOT EXISTS evidence_class text NOT NULL DEFAULT 'inferred';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'entities_evidence_class_check'
    ) THEN
        ALTER TABLE entities ADD CONSTRAINT entities_evidence_class_check
            CHECK (evidence_class IN ('structured', 'syntactic', 'inferred'));
    END IF;
END $$;

COMMENT ON COLUMN entities.evidence_class IS
    'How this entity was established: structured (adapter provenance), '
    'syntactic (unambiguous marker in the text), inferred (dictionary or '
    'keyword lookup). NOT a confidence score -- see TBU-261 and migration 049.';

-- Backfill from entity_type via the same mapping the write path uses.
UPDATE entities
   SET evidence_class = entity_evidence_class(entity_type)
 WHERE evidence_class IS DISTINCT FROM entity_evidence_class(entity_type);

-- ── Write path ────────────────────────────────────────────────────────
-- Unchanged from the shipped version except that the insert stamps the class
-- and the conflict branch keeps the strongest class seen.

CREATE OR REPLACE FUNCTION link_memory_entities(
    p_memory_id uuid,
    p_profile text,
    p_entity_tags text[]
) RETURNS integer
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, extensions
AS $$
DECLARE
    inserted_count integer := 0;
BEGIN
    IF p_entity_tags IS NULL OR array_length(p_entity_tags, 1) IS NULL THEN
        RETURN 0;
    END IF;

    WITH parsed AS (
        SELECT split_part(t, ':', 1) AS et,
               split_part(t, ':', 2) AS cn
        FROM unnest(p_entity_tags) AS t
        WHERE t LIKE '%:%' AND length(split_part(t, ':', 2)) > 0
    ),
    entity_upsert AS (
        INSERT INTO entities (canonical_name, entity_type, mention_count, evidence_class)
        SELECT cn, et, 1, entity_evidence_class(et) FROM parsed
        ON CONFLICT (canonical_name, entity_type) DO UPDATE
            SET mention_count = entities.mention_count + 1,
                evidence_class = CASE
                    WHEN entity_evidence_rank(EXCLUDED.evidence_class)
                       > entity_evidence_rank(entities.evidence_class)
                    THEN EXCLUDED.evidence_class
                    ELSE entities.evidence_class
                END
        RETURNING id
    ),
    edge_insert AS (
        INSERT INTO memory_entities (memory_id, entity_id, profile)
        SELECT p_memory_id, eu.id, p_profile
        FROM entity_upsert eu
        ON CONFLICT (memory_id, entity_id) DO NOTHING
        RETURNING memory_id
    )
    SELECT count(*) INTO inserted_count FROM edge_insert;

    RETURN inserted_count;
END;
$$;

REVOKE EXECUTE ON FUNCTION entity_evidence_class(text) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION entity_evidence_rank(text) FROM PUBLIC;

-- anon/authenticated do not exist on vanilla Postgres, so the Supabase-only
-- half is guarded the same way as migrations 037/041. Without this the two
-- Supabase schema files and a migrated database end up with different grants
-- on these functions, which is precisely the drift TBU-228 was about.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
        RAISE NOTICE
            'anon role not found -- skipping anon/authenticated REVOKE on '
            'entity_evidence_class/_rank (non-Supabase install)';
        RETURN;
    END IF;

    EXECUTE 'REVOKE EXECUTE ON FUNCTION entity_evidence_class(text) '
            'FROM anon, authenticated';
    EXECUTE 'REVOKE EXECUTE ON FUNCTION entity_evidence_rank(text) '
            'FROM anon, authenticated';
END $$;
