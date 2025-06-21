-- Enable Row Level Security
ALTER TABLE auth.users ENABLE ROW LEVEL SECURITY;

-- Create projects table
CREATE TABLE IF NOT EXISTS projects (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Create project_data table
CREATE TABLE IF NOT EXISTS project_data (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    data BYTEA,  -- Encrypted data
    field_types BYTEA,  -- Encrypted field types
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Create dashboards table
CREATE TABLE IF NOT EXISTS dashboards (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    charts JSONB,  -- Store chart configurations
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL
);

-- Create RLS policies for projects
CREATE POLICY "Users can view their own projects"
    ON projects FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own projects"
    ON projects FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own projects"
    ON projects FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own projects"
    ON projects FOR DELETE
    USING (auth.uid() = user_id);

-- Create RLS policies for project_data
CREATE POLICY "Users can view their own project data"
    ON project_data FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own project data"
    ON project_data FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own project data"
    ON project_data FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own project data"
    ON project_data FOR DELETE
    USING (auth.uid() = user_id);

-- Create RLS policies for dashboards
CREATE POLICY "Users can view their own dashboards"
    ON dashboards FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own dashboards"
    ON dashboards FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own dashboards"
    ON dashboards FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own dashboards"
    ON dashboards FOR DELETE
    USING (auth.uid() = user_id);

-- Enable RLS on all tables
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE project_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE dashboards ENABLE ROW LEVEL SECURITY;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_projects_user_id ON projects(user_id);
CREATE INDEX IF NOT EXISTS idx_project_data_project_id ON project_data(project_id);
CREATE INDEX IF NOT EXISTS idx_project_data_user_id ON project_data(user_id);
CREATE INDEX IF NOT EXISTS idx_dashboards_project_id ON dashboards(project_id);
CREATE INDEX IF NOT EXISTS idx_dashboards_user_id ON dashboards(user_id);

-- AI Feedback System Tables
-- This table stores user feedback for AI learning

CREATE TABLE IF NOT EXISTS ai_feedback (
    id SERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    feedback_type TEXT NOT NULL, -- 'chart_rating', 'train_ai', 'business_context'
    content JSONB NOT NULL, -- Stores the full feedback data
    dataset_info JSONB, -- Stores dataset metadata for matching similar datasets
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_ai_feedback_user_id ON ai_feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_ai_feedback_type ON ai_feedback(feedback_type);
CREATE INDEX IF NOT EXISTS idx_ai_feedback_created_at ON ai_feedback(created_at);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_ai_feedback_updated_at 
    BEFORE UPDATE ON ai_feedback 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Enable Row Level Security (RLS)
ALTER TABLE ai_feedback ENABLE ROW LEVEL SECURITY;

-- Create policy to allow users to see only their own feedback
CREATE POLICY "Users can view their own feedback" ON ai_feedback
    FOR SELECT USING (auth.uid()::text = user_id);

-- Create policy to allow users to insert their own feedback
CREATE POLICY "Users can insert their own feedback" ON ai_feedback
    FOR INSERT WITH CHECK (auth.uid()::text = user_id);

-- Create policy to allow users to update their own feedback
CREATE POLICY "Users can update their own feedback" ON ai_feedback
    FOR UPDATE USING (auth.uid()::text = user_id);

-- Create a view for aggregated feedback statistics
CREATE OR REPLACE VIEW ai_feedback_stats AS
SELECT 
    user_id,
    feedback_type,
    COUNT(*) as feedback_count,
    AVG((content->>'rating')::numeric) as avg_rating,
    MIN(created_at) as first_feedback,
    MAX(created_at) as last_feedback
FROM ai_feedback
WHERE content->>'rating' IS NOT NULL
GROUP BY user_id, feedback_type;

-- Grant necessary permissions
GRANT SELECT, INSERT, UPDATE ON ai_feedback TO authenticated;
GRANT SELECT ON ai_feedback_stats TO authenticated; 