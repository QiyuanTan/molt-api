/**
 * Database Migration Runner
 * 
 * Executes SQL migrations from scripts/ directory
 * Tracks applied migrations to ensure idempotent re-runs
 * 
 * Usage: node scripts/migrate.js
 */

const fs = require('fs');
const path = require('path');
const { Pool } = require('pg');
require('dotenv').config({ path: path.join(__dirname, '../.env') });

// Configuration
const DATABASE_URL = process.env.DATABASE_URL;
const NODE_ENV = process.env.NODE_ENV || 'development';
const SSL_ENABLED = NODE_ENV === 'production';

/**
 * Parse and validate DATABASE_URL
 */
function validateConfig() {
  if (!DATABASE_URL) {
    console.error('Error: DATABASE_URL environment variable is not set');
    process.exit(1);
  }
}

/**
 * Create database connection pool
 */
function createPool() {
  const pool = new Pool({
    connectionString: DATABASE_URL,
    ssl: SSL_ENABLED ? { rejectUnauthorized: false } : false,
    max: 1,
    idleTimeoutMillis: 5000,
    connectionTimeoutMillis: 5000
  });

  pool.on('error', (err) => {
    console.error('Database pool error:', err.message);
  });

  return pool;
}

/**
 * Initialize migrations tracking table
 */
async function initializeMigrationsTable(client) {
  const createTableSQL = `
    CREATE TABLE IF NOT EXISTS schema_migrations (
      id SERIAL PRIMARY KEY,
      name VARCHAR(255) NOT NULL UNIQUE,
      applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
      checksum VARCHAR(64)
    );
  `;

  try {
    await client.query(createTableSQL);
    console.log('✓ Migrations table initialized');
  } catch (error) {
    throw new Error(`Failed to initialize migrations table: ${error.message}`);
  }
}

/**
 * Get list of applied migrations
 */
async function getAppliedMigrations(client) {
  try {
    const result = await client.query(
      'SELECT name, checksum FROM schema_migrations ORDER BY name'
    );
    return result.rows;
  } catch (error) {
    throw new Error(`Failed to fetch applied migrations: ${error.message}`);
  }
}

/**
 * Split SQL file into individual statements
 * Handles comments and multi-line statements
 */
function parseSQLStatements(sqlContent) {
  // Remove SQL comments
  let cleaned = sqlContent
    .split('\n')
    .filter(line => !line.trim().startsWith('--'))
    .join('\n');

  // Split by semicolon and filter empty statements
  const statements = cleaned
    .split(';')
    .map(stmt => stmt.trim())
    .filter(stmt => stmt.length > 0);

  return statements;
}

/**
 * Calculate simple checksum for SQL content
 * Uses basic hash to detect if migration file has been modified
 */
function calculateChecksum(content) {
  const crypto = require('crypto');
  return crypto.createHash('sha256').update(content).digest('hex');
}

/**
 * Read migration files from scripts directory
 */
function getMigrationFiles() {
  const migrationsDir = __dirname;
  
  // Currently support schema.sql; can expand to timestamped files later
  const migrationFile = path.join(migrationsDir, 'schema.sql');
  
  if (!fs.existsSync(migrationFile)) {
    throw new Error(`Migration file not found: ${migrationFile}`);
  }

  return [{
    name: 'schema.sql',
    path: migrationFile
  }];
}

/**
 * Read and prepare migrations to apply
 */
function prepareMigrations(migrationFiles, appliedMigrations) {
  const appliedMap = new Map(
    appliedMigrations.map(m => [m.name, m.checksum])
  );

  const pending = [];

  for (const file of migrationFiles) {
    const content = fs.readFileSync(file.path, 'utf-8');
    const checksum = calculateChecksum(content);
    const applied = appliedMap.get(file.name);

    if (!applied) {
      // Migration not applied yet
      pending.push({
        name: file.name,
        content,
        checksum,
        isNew: true
      });
    } else if (applied !== checksum) {
      // Migration exists but file has changed (potential problem)
      console.warn(`⚠ Warning: Migration "${file.name}" has been modified since it was applied`);
      // Don't re-apply to avoid data loss; user should handle manually
    }
  }

  return pending;
}

/**
 * Apply a single migration
 */
async function applyMigration(client, migration) {
  const statements = parseSQLStatements(migration.content);
  
  try {
    for (const statement of statements) {
      await client.query(statement);
    }

    // Record in migrations table
    await client.query(
      'INSERT INTO schema_migrations (name, checksum) VALUES ($1, $2)',
      [migration.name, migration.checksum]
    );

    console.log(`✓ Applied migration: ${migration.name}`);
    return true;
  } catch (error) {
    throw new Error(`Failed to apply migration "${migration.name}": ${error.message}`);
  }
}

/**
 * Main migration runner
 */
async function runMigrations() {
  validateConfig();

  const pool = createPool();
  const client = await pool.connect();

  try {
    await client.query('BEGIN');

    // Initialize migrations table
    await initializeMigrationsTable(client);

    // Get already applied migrations
    const applied = await getAppliedMigrations(client);

    // Get pending migrations
    const migrationFiles = getMigrationFiles();
    const pending = prepareMigrations(migrationFiles, applied);

    if (pending.length === 0) {
      console.log('No pending migrations');
      await client.query('COMMIT');
      return true;
    }

    console.log(`\nFound ${pending.length} pending migration(s):\n`);

    // Apply each migration in transaction
    for (const migration of pending) {
      await applyMigration(client, migration);
    }

    await client.query('COMMIT');

    console.log(`\n✓ Successfully applied ${pending.length} migration(s)`);
    return true;
  } catch (error) {
    await client.query('ROLLBACK');
    console.error('\n✗ Migration failed:', error.message);
    return false;
  } finally {
    client.release();
    await pool.end();
  }
}

/**
 * Entry point
 */
(async () => {
  try {
    console.log('Running database migrations...\n');
    const success = await runMigrations();
    process.exit(success ? 0 : 1);
  } catch (error) {
    console.error('Fatal error:', error.message);
    process.exit(1);
  }
})();
