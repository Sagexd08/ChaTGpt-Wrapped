/**
 * Shared utility functions
 */

export const formatNumber = (num: number): string => {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M';
  }
  if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'K';
  }
  return num.toString();
};

export const formatDuration = (seconds: number): string => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${minutes}m`;
};

export const delay = (ms: number): Promise<void> => {
  return new Promise(resolve => setTimeout(resolve, ms));
};

export const generateId = (): string => {
  return Math.random().toString(36).substring(2) + Date.now().toString(36);
};

export const parseJsonData = (data: string): any => {
  try {
    return JSON.parse(data);
  } catch {
    return null;
  }
};

export const sanitizeText = (text: string): string => {
  return text
    .replace(/[<>]/g, '')
    .trim()
    .substring(0, 500);
};

export const getColorByTopic = (topic: string): string => {
  const colors: Record<string, string> = {
    tech: '#FF006E',
    learning: '#00D9FF',
    career: '#39FF14',
    creative: '#B100FF',
    casual: '#FF9500',
  };
  return colors[topic] || '#00D9FF';
};

export const getColorBySentiment = (sentiment: string): string => {
  const colors: Record<string, string> = {
    positive: '#39FF14',
    neutral: '#00D9FF',
    negative: '#FF006E',
  };
  return colors[sentiment] || '#00D9FF';
};
