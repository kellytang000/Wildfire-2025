"use client";

import React, { useState, useEffect } from 'react';
import { MenuOutlined, CloseOutlined } from '@ant-design/icons';

interface MobileMenuToggleProps {
  onToggle: (isOpen: boolean) => void;
}

const MobileMenuToggle: React.FC<MobileMenuToggleProps> = ({ onToggle }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const checkIsMobile = () => {
      setIsMobile(window.innerWidth <= 768);
    };

    checkIsMobile();
    window.addEventListener('resize', checkIsMobile);
    
    return () => window.removeEventListener('resize', checkIsMobile);
  }, []);

  const handleToggle = () => {
    const newState = !isOpen;
    setIsOpen(newState);
    onToggle(newState);
  };

  if (!isMobile) return null;

  return (
    <button
      className="eva-button eva-button--primary"
      onClick={handleToggle}
      style={{
        position: 'fixed',
        top: 'var(--spacing-md)',
        left: 'var(--spacing-md)',
        zIndex: 1001,
        borderRadius: '50%',
        width: '48px',
        height: '48px',
        padding: 0,
        boxShadow: 'var(--shadow-lg)'
      }}
    >
      {isOpen ? <CloseOutlined /> : <MenuOutlined />}
    </button>
  );
};

export default MobileMenuToggle;
