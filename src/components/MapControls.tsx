"use client";

import { useMapContext, MapTheme } from "@/src/context/map.ctx";

const MapControls = () => {
  const { mapTheme, setMapTheme, mapType, setMapType } = useMapContext();

  const handleOriginalMap = () => {
    setMapType('roadmap');
    setMapTheme('original');
  };

  const handleSatelliteMap = () => {
    setMapType('satellite');
    setMapTheme('original'); // Satellite doesn't need special theme
  };

  const handleDarkView = () => {
    setMapType('roadmap');
    setMapTheme('darkview');
  };

  const isOriginalActive = mapType === 'roadmap' && mapTheme === 'original';
  const isSatelliteActive = mapType === 'satellite';
  const isDarkViewActive = mapTheme === 'darkview' && mapType === 'roadmap';

  return (
    <div style={{
      position: 'fixed',
      top: '20px',
      left: 'calc(var(--sidebar-width) + 20px)',
      zIndex: 1000,
      display: 'flex',
      background: 'rgba(255, 255, 255, 0.95)',
      borderRadius: '8px',
      boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)',
      backdropFilter: 'blur(8px)',
      overflow: 'hidden'
    }}>
      {/* Original Map */}
      <button
        onClick={handleOriginalMap}
        style={{
          padding: '12px 20px',
          border: 'none',
          background: isOriginalActive ? '#F87A53' : 'transparent',
          color: isOriginalActive ? 'white' : '#666',
          fontSize: '14px',
          fontWeight: '500',
          cursor: 'pointer',
          transition: 'all 0.2s ease',
          borderRight: '1px solid rgba(0,0,0,0.1)'
        }}
        onMouseEnter={(e) => {
          if (!isOriginalActive) {
            e.currentTarget.style.background = 'rgba(24, 144, 255, 0.1)';
          }
        }}
        onMouseLeave={(e) => {
          if (!isOriginalActive) {
            e.currentTarget.style.background = 'transparent';
          }
        }}
      >
        Map
      </button>

      {/* Satellite */}
      <button
        onClick={handleSatelliteMap}
        style={{
          padding: '12px 20px',
          border: 'none',
          background: isSatelliteActive ? '#F87A53' : 'transparent',
          color: isSatelliteActive ? 'white' : '#666',
          fontSize: '14px',
          fontWeight: '500',
          cursor: 'pointer',
          transition: 'all 0.2s ease',
          borderRight: '1px solid rgba(0,0,0,0.1)'
        }}
        onMouseEnter={(e) => {
          if (!isSatelliteActive) {
            e.currentTarget.style.background = 'rgba(24, 144, 255, 0.1)';
          }
        }}
        onMouseLeave={(e) => {
          if (!isSatelliteActive) {
            e.currentTarget.style.background = 'transparent';
          }
        }}
      >
        Satellite
      </button>

      {/* DarkView */}
      <button
        onClick={handleDarkView}
        style={{
          padding: '12px 20px',
          border: 'none',
          background: isDarkViewActive ? '#F87A53' : 'transparent',
          color: isDarkViewActive ? 'white' : '#666',
          fontSize: '14px',
          fontWeight: '500',
          cursor: 'pointer',
          transition: 'all 0.2s ease'
        }}
        onMouseEnter={(e) => {
          if (!isDarkViewActive) {
            e.currentTarget.style.background = 'rgba(255, 69, 0, 0.1)';
          }
        }}
        onMouseLeave={(e) => {
          if (!isDarkViewActive) {
            e.currentTarget.style.background = 'transparent';
          }
        }}
      >
        DarkView
      </button>
    </div>
  );
};

export default MapControls;
