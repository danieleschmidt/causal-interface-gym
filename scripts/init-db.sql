-- Database initialization script for Causal Interface Gym

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create database user for application
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'causal_gym_app') THEN
        CREATE ROLE causal_gym_app LOGIN PASSWORD 'app_password_change_me';
    END IF;
END
$$;

-- Grant necessary permissions
GRANT CONNECT ON DATABASE causal_gym_prod TO causal_gym_app;
GRANT USAGE ON SCHEMA public TO causal_gym_app;
GRANT CREATE ON SCHEMA public TO causal_gym_app;

-- Create tables (will be created by application, but we can prepare)
-- This ensures the schema is ready

-- Performance optimizations
-- Increase shared_buffers for better performance
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Reload configuration
SELECT pg_reload_conf();