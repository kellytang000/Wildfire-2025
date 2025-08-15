"use client";

import React from 'react';
import { BarChartOutlined } from '@ant-design/icons';

interface StatsCardProps {
  title?: string;
  subtitle?: string;
  stats?: Array<{
    label: string;
    value: string | number;
    trend?: 'up' | 'down' | 'stable';
  }>;
}

const StatsCard: React.FC<StatsCardProps> = ({ 
  title = "Emergency Statistics",
  stats = [
    { label: "Active Shelters", value: 10, trend: 'up' },
    { label: "Evacuation Routes", value: 8, trend: 'stable' },
    { label: "People Evacuated", value: "1,247", trend: 'up' },
    { label: "Response Time", value: "12 min", trend: 'down' }
  ]
}) => {
  const getTrendIcon = (trend?: 'up' | 'down' | 'stable') => {
    switch (trend) {
      case 'up': return '↗';
      case 'down': return '↘';
      default: return '→';
    }
  };

  const getTrendColor = (trend?: 'up' | 'down' | 'stable') => {
    switch (trend) {
      case 'up': return 'var(--color-success)';
      case 'down': return 'var(--color-danger)';
      default: return 'var(--color-text-muted)';
    }
  };

  return (
    <div className="eva-card">
      <div className="eva-flex eva-flex--between" style={{ marginBottom: 'var(--spacing-md)' }}>
        <div>
          <h3 className="eva-heading">{title}</h3>
          <div className="eva-badge eva-badge--warning">
            LIVE DATA
          </div>
        </div>
        <BarChartOutlined style={{ color: 'var(--color-text-secondary)', fontSize: '20px' }} />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--spacing-lg)' }}>
        {stats.map((stat, index) => (
          <div key={index}>
            <div className="eva-text eva-text--small eva-text--muted" style={{ marginBottom: 'var(--spacing-xs)' }}>
              {stat.label}
            </div>
            <div className="eva-flex" style={{ gap: 'var(--spacing-xs)', alignItems: 'baseline' }}>
              <span style={{ 
                fontSize: 'var(--font-size-lg)', 
                fontWeight: 600, 
                color: 'var(--color-text-primary)' 
              }}>
                {stat.value}
              </span>
              {stat.trend && (
                <span style={{ 
                  fontSize: 'var(--font-size-xs)', 
                  color: getTrendColor(stat.trend) 
                }}>
                  {getTrendIcon(stat.trend)}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default StatsCard;
