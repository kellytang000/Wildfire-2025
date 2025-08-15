"use client";

import React from 'react';
import { EnvironmentOutlined, FireOutlined, ExclamationCircleOutlined, DownOutlined } from '@ant-design/icons';
import { Modal, Form, Input, Dropdown } from 'antd';
import { useBoolean } from 'ahooks';
import { useEscape } from '@/src/actions/closure.hook';
import { useMapContext } from '@/src/context/map.ctx';

interface LocationInfoCardProps {
  address?: string;
  region?: string;
  description?: string;
  coordinates?: {
    lat: number;
    lng: number;
  };
  fireLevel?: 'Low' | 'Moderate' | 'High' | 'Extreme';
  riskScore?: number;
  riskDescription?: string;
  currentPosition?: {
    lat: number;
    lng: number;
  };
}

const LocationInfoCard: React.FC<LocationInfoCardProps> = ({ 
  address = "Current Location",
//   region = "Emergency Zone",
//   description = "This area is currently under wildfire monitoring with evacuation routes available.",
  coordinates,
  fireLevel = 'Moderate',
  riskScore = 4,
  riskDescription = 'This area has increased wildfire risk due to current weather conditions.',
  currentPosition
}) => {
  const [currentLocationName, setCurrentLocationName] = React.useState('Alice Lake Rd, Brackendale, BC');
  const [currentCoordinates, setCurrentCoordinates] = React.useState({ lat: 49.78888350, lng: -123.1318467 });
  const [isModalOpen, setIsModalOpen] = useBoolean();
  const { createEscape, loading: loadingEscape } = useEscape();
  const { route, setRoute, simulatedUserLocation, setSimulatedUserLocation } = useMapContext();
  const [form] = Form.useForm();
  
  const locationOptions = [
    {
      name: 'Alice Lake Rd, Brackendale, BC',
      fullAddress: 'Alice Lake Rd, Brackendale, BC V0N 1H0',
      coordinates: { lat: 49.78888350, lng: -123.1318467 }
    },
    {
      name: '325 Toyopa Dr, Pacific Palisades, CA',
      fullAddress: '325 Toyopa Dr, Pacific Palisades, CA 90272, USA',
      coordinates: { lat: 34.0339439, lng: -118.5208065 }
    }
  ];

  // Initialize with first location but don't change map center
  React.useEffect(() => {
    if (locationOptions.length > 0) {
      setCurrentLocationName(locationOptions[0].name);
      setCurrentCoordinates(locationOptions[0].coordinates);
      // Initialize simulated user location
      setSimulatedUserLocation({
        lat: locationOptions[0].coordinates.lat,
        lng: locationOptions[0].coordinates.lng,
        address: locationOptions[0].name
      });
    }
  }, [setSimulatedUserLocation]);
  
  const handleLocationChange = (locationOption: typeof locationOptions[0]) => {
    setCurrentLocationName(locationOption.name);
    setCurrentCoordinates(locationOption.coordinates);
    // Update simulated user location on the map
    setSimulatedUserLocation({
      lat: locationOption.coordinates.lat,
      lng: locationOption.coordinates.lng,
      address: locationOption.name
    });
  };

  const handlePlanEvacuation = async () => {
    try {
      const values = await form.validateFields();

      // 确保坐标值是数字类型
      const params = {
        latitude: parseFloat(values.latitude),
        longitude: parseFloat(values.longitude)
      };

      // 验证坐标值是否有效
      if (isNaN(params.latitude) || isNaN(params.longitude)) {
        console.error('Invalid coordinates:', values);
        return;
      }

      console.log('Planning evacuation route with params:', params);

      // 先清除旧路径
      setRoute(null);

      const res = await createEscape(params);
      console.log('Evacuation route response:', res);

      if (res) {
        setRoute(res);
      }

      setIsModalOpen.setFalse();
    } catch (error) {
      console.error('Error planning evacuation route:', error);
    }
  };

  // 打开模态框时自动填充当前地址的坐标
  const handleOpenModal = () => {
    form.setFieldsValue({
      latitude: currentCoordinates.lat.toString(),
      longitude: currentCoordinates.lng.toString()
    });
    setIsModalOpen.setTrue();
  };
  const getRiskColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'low': return '#10b981';
      case 'moderate': return '#f59e0b';
      case 'high': return '#dc2626';
      case 'extreme': return '#991b1b';
      default: return '#f59e0b';
    }
  };

  const getRiskBgColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'low': return '#d1fae5';
      case 'moderate': return '#fef3c7';
      case 'high': return '#fee2e2';
      case 'extreme': return '#fecaca';
      default: return '#fef3c7';
    }
  };
  return (
    <div className="eva-card">
      {/* Header with back arrow and actions */}
      <div className="eva-flex eva-flex--between" style={{ marginBottom: '8px' }}>
        <div className="eva-flex" style={{ gap: 'var(--spacing-sm)' }}>
          {/* <button style={{ 
            background: 'none', 
            border: 'none', 
            fontSize: '18px', 
            color: 'var(--color-text-secondary)',
            cursor: 'pointer',
            padding: 0
          }}>
            ←
          </button> */}
          {/* <span className="eva-text eva-text--small" style={{ color: 'var(--color-text-secondary)' }}>
            Location Info
          </span> */}
        </div>
      </div>

      {/* Location Title and Priority Badge */}
      <div style={{ marginBottom: '12px' }}>
        <div className="eva-flex" style={{ gap: 'var(--spacing-xs)', marginBottom: 'var(--spacing-xs)' }}>
          <EnvironmentOutlined style={{ color: 'var(--color-text-secondary)', fontSize: '16px' }} />
          <span className="eva-text eva-text--small eva-text--muted">
            {currentPosition ? `${currentPosition.lat.toFixed(6)}, ${currentPosition.lng.toFixed(6)}` : `${currentCoordinates.lat.toFixed(6)}, ${currentCoordinates.lng.toFixed(6)}`}
          </span>
        </div>
        
        <div className="eva-flex eva-flex--between" style={{ alignItems: 'center', marginBottom: 'var(--spacing-sm)' }}>
          <h1 className="eva-title" style={{ fontSize: '18px' }}>
            {currentLocationName}
          </h1>
          <Dropdown menu={{ items: locationOptions.map(location => ({
            key: location.name,
            label: <span style={{ fontSize: '11px' }}>{location.name}</span>,
            onClick: () => handleLocationChange(location)
          })) }} trigger={['click']}>
            <button 
              style={{ 
                background: 'none', 
                border: '1px solid var(--color-border)', 
                borderRadius: '6px',
                fontSize: '12px', 
                color: 'var(--color-text-secondary)',
                cursor: 'pointer',
                padding: '4px 8px',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                transition: 'all 0.2s ease'
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.borderColor = 'var(--color-primary)';
                e.currentTarget.style.color = 'var(--color-primary)';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.borderColor = 'var(--color-border)';
                e.currentTarget.style.color = 'var(--color-text-secondary)';
              }}
            >
              {/* <DownOutlined style={{ fontSize: '10px' }} /> */}
              Switch
              <DownOutlined style={{ fontSize: '8px' }} />
            </button>
          </Dropdown>
        </div>
        
        <div className="eva-flex" style={{ gap: 'var(--spacing-sm)', marginBottom: '8px' }}>
          <div 
            className="eva-badge" 
            style={{ 
              backgroundColor: getRiskBgColor('high'), 
              color: getRiskColor('high'),
              border: `1px solid ${getRiskColor('high')}20`
            }}
          >
            HIGH PRIORITY
          </div>
          <div 
            className="eva-badge" 
            style={{ 
              backgroundColor: getRiskBgColor('high'), 
              color: getRiskColor('high'),
              border: `1px solid ${getRiskColor('high')}20`
            }}
          >
            HIGH RISK
          </div>
        </div>
      </div>

      {/*  Section */}
      <div style={{ marginBottom: '12px' }}>
        {/* <div className="eva-flex eva-flex--between" style={{ marginBottom: 'var(--spacing-md)' }}>
          <h3 className="eva-heading" style={{ fontSize: '16px' }}>Environmental Risk</h3>
          <div className="eva-flex" style={{ gap: 'var(--spacing-xs)' }}>
            <FireOutlined style={{ color: getRiskColor(fireLevel), fontSize: '16px' }} />
            <ExclamationCircleOutlined style={{ color: getRiskColor(fireLevel), fontSize: '16px' }} />
          </div>
        </div> */}

        <div className="eva-flex" style={{ gap: 'var(--spacing-sm)', marginBottom: '8px' }}>
          <div style={{ 
            width: '20px', 
            height: '20px', 
            borderRadius: '4px',
            background: getRiskColor(fireLevel),
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <FireOutlined style={{ color: 'white', fontSize: '12px' }} />
          </div>
          <div style={{ flex: 1 }}>
            <div className="eva-text" style={{ fontWeight: 500, color: 'var(--color-text-primary)', marginBottom: 'var(--spacing-xs)' }}>
              Fire Factor: <span style={{ color: getRiskColor(fireLevel) }}>{fireLevel}</span>
            </div>
            <div className="eva-text eva-text--small eva-text--muted">
              {riskDescription}
            </div>
          </div>
          <div style={{
            width: '40px',
            height: '40px',
            border: '1px solid var(--color-border)',
            borderRadius: 'var(--radius-md)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'var(--color-background)',
            fontWeight: 'bold',
            color: getRiskColor(fireLevel)
          }}>
            {riskScore}
          </div>
        </div>
      </div>

      {/* Your Location Section
      <div style={{ marginBottom: 'var(--spacing-lg)' }}>
        <h3 className="eva-heading" style={{ fontSize: '16px', marginBottom: 'var(--spacing-md)' }}>Your Location</h3>
        
        <div style={{ 
          backgroundColor: 'var(--color-background-light)',
          padding: 'var(--spacing-md)',
          borderRadius: 'var(--border-radius)',
          marginBottom: 'var(--spacing-md)'
        }}>
          <div className="eva-text" style={{ fontWeight: 500, marginBottom: 'var(--spacing-sm)' }}>
            Current Position:
          </div>
          <div className="eva-text eva-text--small" style={{ marginBottom: 'var(--spacing-xs)' }}>
            Latitude: {currentPosition ? currentPosition.lat.toFixed(6) : '34.052200'}
          </div>
          <div className="eva-text eva-text--small" style={{ marginBottom: 'var(--spacing-md)' }}>
            Longitude: {currentPosition ? currentPosition.lng.toFixed(6) : '-118.243700'}
          </div>
          
          <button 
            className="eva-button eva-button--secondary" 
            style={{ 
              fontSize: 'var(--font-size-sm)',
              padding: 'var(--spacing-xs) var(--spacing-sm)',
              height: 'auto'
            }}
            onClick={() => {
              // Request GPS location
              if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(
                  (position) => {
                    console.log('GPS location:', position.coords.latitude, position.coords.longitude);
                    // In a real app, this would update the location state
                  },
                  (error) => {
                    console.error('Location error:', error);
                  }
                );
              }
            }}
          >
            Update Location
          </button>
        </div>
      </div> */}

      {/* Action Button - Plan Evacuation Route */}
      <div style={{ marginTop: 'var(--spacing-lg)' }}>
        <button 
          className="eva-button eva-button--primary" 
          onClick={handleOpenModal}
          style={{ 
            width: '100%', 
            fontSize: '14px',
            padding: '12px 16px',
            fontWeight: '600',
            backgroundColor: '#31511E', // 固定背景色（蓝色，可自定义）
            color: '#fff',              // 固定文字色（白色）
            border: 'none'              // 去除边框
          }}
        >
          Plan Evacuation Route
        </button>
        
        {route && (
          <div style={{ 
            marginTop: 'var(--spacing-lg)',
            padding: 'var(--spacing-lg)',
            border: '2px solid var(--color-primary)',
            borderRadius: 'var(--border-radius)',
            background: 'linear-gradient(135deg, var(--color-background-light) 0%, var(--color-background) 100%)',
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
          }}>
            <div style={{ marginBottom: 'var(--spacing-md)' }}>
              <div className="eva-text" style={{ 
                color: 'var(--color-primary)', 
                fontWeight: 'bold',
                fontSize: '16px',
                marginBottom: '8px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}>
                🗺️ Route Planned Successfully
              </div>
              <div className="eva-text" style={{ 
                color: 'var(--color-text-primary)',
                lineHeight: '1.5',
                fontSize: '15px',
                fontWeight: '600'
              }}>
                {route.n_points} waypoints • {Math.round(route.cost_s / 60)} min ({route.cost_s}s)
              </div>
            </div>
            <button 
              className="eva-button eva-button--ghost"
              onClick={() => setRoute(null)}
              style={{
                fontSize: '14px',
                padding: '12px 16px',
                background: 'var(--color-background)',
                border: '1px solid var(--color-border)',
                fontWeight: '600',
                width: '100%',
                borderRadius: 'var(--border-radius)',
                transition: 'all 0.2s ease'
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.background = 'var(--color-primary)';
                e.currentTarget.style.color = 'white';
                e.currentTarget.style.borderColor = 'var(--color-primary)';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.background = 'var(--color-background)';
                e.currentTarget.style.color = 'var(--color-text-primary)';
                e.currentTarget.style.borderColor = 'var(--color-border)';
              }}
            >
              Clear Route
            </button>
          </div>
        )}
      </div>

      {/* Plan Evacuation Route Modal */}
      <Modal
        title="Plan Evacuation Route"
        open={isModalOpen}
        onCancel={setIsModalOpen.setFalse}
        onOk={handlePlanEvacuation}
        okButtonProps={{ loading: loadingEscape }}
      >
        <Form form={form} layout="vertical">
          <Form.Item required label="Latitude" name="latitude" rules={[{ required: true }]}>
            <Input placeholder="Enter latitude" />
          </Form.Item>
          <Form.Item required label="Longitude" name="longitude" rules={[{ required: true }]}>
            <Input placeholder="Enter longitude" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default LocationInfoCard;
