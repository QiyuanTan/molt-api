-- Migration: Increase VARCHAR field sizes for agents and submolts

-- Alter agents table
ALTER TABLE IF EXISTS agents
    ALTER COLUMN name SET DATA TYPE VARCHAR(128);

ALTER TABLE IF EXISTS agents
    ALTER COLUMN display_name SET DATA TYPE VARCHAR(128);

ALTER TABLE IF EXISTS agents
    ALTER COLUMN status SET DATA TYPE VARCHAR(64);

-- Alter submolts table
ALTER TABLE IF EXISTS submolts
    ALTER COLUMN name SET DATA TYPE VARCHAR(128);

ALTER TABLE IF EXISTS submolts
    ALTER COLUMN display_name SET DATA TYPE VARCHAR(128);

-- Success message (visible in psql output)
SELECT 'Schema migration completed successfully' AS status;
