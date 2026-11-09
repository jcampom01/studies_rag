-- Runs once, when the database is created for the first time (empty data volume).
-- Extensions need elevated privileges: they are set up here, not by the application.
CREATE EXTENSION IF NOT EXISTS vector;
