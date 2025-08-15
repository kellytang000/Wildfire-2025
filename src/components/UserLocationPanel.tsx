"use client";

import React, { useState } from 'react';
import { Card, Form, InputNumber, Button, message, Space, Typography, Divider } from 'antd';
import { EnvironmentOutlined, EditOutlined, SaveOutlined } from '@ant-design/icons';
import { useCurrentUserLocation } from '../actions/user-location.hook';

const { Title, Text } = Typography;

interface LocationFormData {
  lat: number;
  lng: number;
}

export const UserLocationPanel: React.FC = () => {
  const { currentLocation, loading, updateLocation } = useCurrentUserLocation();
  const [form] = Form.useForm<LocationFormData>();
  const [isEditing, setIsEditing] = useState(false);

  // Helper function to check if coordinates are valid
  const isValidCoordinate = (lat: number | null, lng: number | null): boolean => {
    return typeof lat === 'number' && 
           typeof lng === 'number' && 
           !isNaN(lat) && 
           !isNaN(lng) && 
           lat >= -90 && 
           lat <= 90 && 
           lng >= -180 && 
           lng <= 180;
  };

  const handleUpdateLocation = async (values: LocationFormData) => {
    try {
      await updateLocation(values.lat, values.lng);
      message.success('Location updated successfully!');
      setIsEditing(false);
    } catch (error) {
      message.error('Failed to update location. Please try again.');
    }
  };

  const handleEdit = () => {
    if (currentLocation && isValidCoordinate(currentLocation.lat, currentLocation.lng)) {
      form.setFieldsValue({
        lat: currentLocation.lat as number,
        lng: currentLocation.lng as number,
      });
    }
    setIsEditing(true);
  };

  const handleCancel = () => {
    form.resetFields();
    setIsEditing(false);
  };

  return (
    <div className="eva-card">
      <div className="eva-flex eva-flex--between" style={{ marginBottom: 'var(--spacing-md)' }}>
        <h3 className="eva-heading">Your Location</h3>
        <EnvironmentOutlined style={{ color: 'var(--color-text-secondary)', fontSize: '16px' }} />
      </div>

      <div style={{ marginBottom: 16 }}>
        <Text strong>Current Position:</Text>
        <br />
        {currentLocation && isValidCoordinate(currentLocation.lat, currentLocation.lng) ? (
          <>
            <Text>Latitude: {(currentLocation.lat as number).toFixed(6)}</Text>
            <br />
            <Text>Longitude: {(currentLocation.lng as number).toFixed(6)}</Text>
          </>
        ) : (
          <Text type="secondary">No location set</Text>
        )}
      </div>

      <Divider style={{ margin: '12px 0' }} />

      {!isEditing ? (
        <Button
          type="primary"
          icon={<EditOutlined />}
          onClick={handleEdit}
          block
          className="eva-button"
        >
          Update Location
        </Button>
      ) : (
        <Form
          form={form}
          layout="vertical"
          onFinish={handleUpdateLocation}
          size="small"
        >
          <Form.Item
            name="lat"
            label="Latitude"
            rules={[
              { required: true, message: 'Please enter latitude' },
              {
                type: 'number',
                min: -90,
                max: 90,
                message: 'Latitude must be between -90 and 90',
              },
            ]}
          >
            <InputNumber
              step={0.000001}
              precision={6}
              placeholder="e.g., 47.6062"
              disabled={loading}
              style={{ width: '100%' }}
            />
          </Form.Item>

          <Form.Item
            name="lng"
            label="Longitude"
            rules={[
              { required: true, message: 'Please enter longitude' },
              {
                type: 'number',
                min: -180,
                max: 180,
                message: 'Longitude must be between -180 and 180',
              },
            ]}
          >
            <InputNumber
              step={0.000001}
              precision={6}
              placeholder="e.g., -122.3321"
              disabled={loading}
              style={{ width: '100%' }}
            />
          </Form.Item>

          <Form.Item style={{ marginBottom: 0 }}>
            <Space>
              <Button
                type="primary"
                htmlType="submit"
                icon={<SaveOutlined />}
                loading={loading}
                className="eva-button"
              >
                Save
              </Button>
              <Button onClick={handleCancel} disabled={loading} className="eva-button">
                Cancel
              </Button>
            </Space>
          </Form.Item>
        </Form>
      )}

      <div style={{ marginTop: 12, fontSize: '12px', color: '#666' }}>
        <Text type="secondary">
          Tip: You can find coordinates using Google Maps or GPS apps
        </Text>
      </div>
    </div>
  );
};
