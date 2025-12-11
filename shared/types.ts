/**
 * Shared types for ChatGPT Wrapped
 */

export interface ChatMessage {
  id?: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp?: string;
}

export interface AnalysisResult {
  status: 'processing' | 'success' | 'failed';
  metrics?: MetricsData;
  sentiment?: SentimentAnalysis;
  topics?: TopicsAnalysis;
  moments?: ChatMoment[];
  achievements?: Achievement[];
  metadata?: Record<string, any>;
  error?: string;
}

export interface MetricsData {
  message_counts: {
    total: number;
    user: number;
    assistant: number;
  };
  word_counts: {
    total_words: number;
    unique_words: number;
    average_message_length: number;
  };
  streaks: {
    longest_user_streak: number;
    average_exchange_length: number;
  };
  activity: {
    most_active_hour: string;
    peak_activity: number;
    message_length_stats: {
      min: number;
      max: number;
      average: number;
    };
  };
}

export interface SentimentAnalysis {
  overall_sentiment: number;
  sentiment_label: 'negative' | 'neutral' | 'positive';
  persona: string;
  emotion_breakdown: {
    positive: number;
    neutral: number;
    negative: number;
  };
}

export interface TopicsAnalysis {
  top_topics: Array<{ name: string; count: number }>;
  topic_distribution: Record<string, number>;
  primary_topic: string;
  topic_diversity: number;
}

export interface ChatMoment {
  id: string;
  content: string;
  type: 'funny' | 'unhinged' | 'repeated' | 'wholesome';
  timestamp?: string;
}

export interface Achievement {
  id: string;
  name: string;
  description: string;
  icon?: string;
  unlocked: boolean;
  progress?: number;
}

export interface WrappedSlide {
  id: string;
  title: string;
  type: 'cover' | 'metrics' | 'mood' | 'moments' | 'topics' | 'achievements' | 'final';
  content: Record<string, any>;
  order: number;
}

export interface WrappedSession {
  id: string;
  created_at: string;
  data: AnalysisResult;
  slides: WrappedSlide[];
  exported: boolean;
}
