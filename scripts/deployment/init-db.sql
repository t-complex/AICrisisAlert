-- Database initialization script for AICrisisAlert
-- This script creates the necessary tables and indexes

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create crisis_classifications table
CREATE TABLE IF NOT EXISTS crisis_classifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    text TEXT NOT NULL,
    predicted_class VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    class_probabilities JSONB NOT NULL,
    processing_time_ms DECIMAL(10,2) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    source VARCHAR(50),
    location VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on predicted_class for faster queries
CREATE INDEX IF NOT EXISTS idx_crisis_classifications_class ON crisis_classifications(predicted_class);

-- Create index on confidence for filtering high-confidence predictions
CREATE INDEX IF NOT EXISTS idx_crisis_classifications_confidence ON crisis_classifications(confidence);

-- Create index on created_at for time-based queries
CREATE INDEX IF NOT EXISTS idx_crisis_classifications_created_at ON crisis_classifications(created_at);

-- Create index on text for full-text search
CREATE INDEX IF NOT EXISTS idx_crisis_classifications_text_gin ON crisis_classifications USING gin(to_tsvector('english', text));

-- Create emergency_alerts table
CREATE TABLE IF NOT EXISTS emergency_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    classification_id UUID REFERENCES crisis_classifications(id),
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    message TEXT NOT NULL,
    location VARCHAR(255),
    coordinates POINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on alert_type and severity
CREATE INDEX IF NOT EXISTS idx_emergency_alerts_type_severity ON emergency_alerts(alert_type, severity);

-- Create index on status for filtering
CREATE INDEX IF NOT EXISTS idx_emergency_alerts_status ON emergency_alerts(status);

-- Create index on coordinates for spatial queries
CREATE INDEX IF NOT EXISTS idx_emergency_alerts_coordinates ON emergency_alerts USING gist(coordinates);

-- Create model_performance table
CREATE TABLE IF NOT EXISTS model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    accuracy DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    precision DECIMAL(5,4),
    recall DECIMAL(5,4),
    test_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on model_version and test_date
CREATE INDEX IF NOT EXISTS idx_model_performance_version_date ON model_performance(model_version, test_date);

-- Create api_requests table for monitoring
CREATE TABLE IF NOT EXISTS api_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id VARCHAR(50) UNIQUE NOT NULL,
    method VARCHAR(10) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    status_code INTEGER NOT NULL,
    processing_time_ms DECIMAL(10,2) NOT NULL,
    client_ip INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on request_id for lookups
CREATE INDEX IF NOT EXISTS idx_api_requests_request_id ON api_requests(request_id);

-- Create index on created_at for time-based analysis
CREATE INDEX IF NOT EXISTS idx_api_requests_created_at ON api_requests(created_at);

-- Create index on status_code for error monitoring
CREATE INDEX IF NOT EXISTS idx_api_requests_status_code ON api_requests(status_code);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers to automatically update updated_at
CREATE TRIGGER update_crisis_classifications_updated_at 
    BEFORE UPDATE ON crisis_classifications 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_emergency_alerts_updated_at 
    BEFORE UPDATE ON emergency_alerts 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create view for recent high-confidence classifications
CREATE OR REPLACE VIEW recent_high_confidence_classifications AS
SELECT 
    id,
    text,
    predicted_class,
    confidence,
    source,
    location,
    created_at
FROM crisis_classifications 
WHERE confidence >= 0.8 
AND created_at >= NOW() - INTERVAL '24 hours'
ORDER BY created_at DESC;

-- Create view for emergency alerts summary
CREATE OR REPLACE VIEW emergency_alerts_summary AS
SELECT 
    alert_type,
    severity,
    status,
    COUNT(*) as count,
    MIN(created_at) as first_alert,
    MAX(created_at) as last_alert
FROM emergency_alerts 
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY alert_type, severity, status
ORDER BY count DESC;

-- Insert sample data for testing (optional)
INSERT INTO model_performance (model_version, model_type, accuracy, f1_score, precision, recall) 
VALUES ('1.0.0', 'bertweet', 0.84, 0.78, 0.82, 0.76)
ON CONFLICT DO NOTHING;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO crisis_alert_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO crisis_alert_user; 