"use client";

import React, { useState } from 'react';
import { Form, InputNumber, Button, message, Space } from 'antd';
import { EnvironmentOutlined, EditOutlined, SaveOutlined } from '@ant-design/icons';
import { useCurrentUserLocation } from '../actions/user-location.hook';

interface LocationFormData {
  lat: number;
  lng: number;
}

export const UserLocationPanel: React.FC = () => {
  const { currentLocation, loading, updateLocation } = useCurrentUserLocation();
  const [form] = Form.useForm<LocationFormData>();
  const [isEditing, setIsEditing] = useState(false);

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
    if (currentLocation?.lat && currentLocation?.lng) {
      form.setFieldsValue({
        lat: currentLocation.lat,
        lng: currentLocation.lng,
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

      <div style={{ marginBottom: 'var(--spacing-md)' }}>
        <div className="eva-text" style={{ fontWeight: 500, marginBottom: 'var(--spacing-xs)' }}>
          Current Position:
        </div>
        {currentLocation?.lat && currentLocation?.lng ? (
          <div className="eva-stack eva-stack--sm">
            <div className="eva-text eva-text--small eva-text--muted">
              Latitude: {currentLocation.lat.toFixed(6)}
            </div>
            <div className="eva-text eva-text--small eva-text--muted">
              Longitude: {currentLocation.lng.toFixed(6)}
            </div>
          </div>
        ) : (
          <div className="eva-text eva-text--muted">
            Location not set
          </div>
        )}
      </div>

      {!isEditing ? (
        <button
          className="eva-button eva-button--secondary"
          onClick={handleEdit}
          disabled={loading}
        >
          <EditOutlined /> Update Location
        </button>
      ) : (
        <Form
          form={form}
          layout="vertical"
          onFinish={handleUpdateLocation}
          style={{ marginTop: 'var(--spacing-md)' }}
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
              className="eva-input"
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
              className="eva-input"
            />
          </Form.Item>
          <Form.Item style={{ marginBottom: 0 }}>
            <Space>
              <button
                type="submit"
                className="eva-button eva-button--primary eva-button--small"
                disabled={loading}
              >
                <SaveOutlined /> Save
              </button>
              <button
                type="button"
                className="eva-button eva-button--ghost eva-button--small"
                onClick={handleCancel}
                disabled={loading}
              >
                Cancel
              </button>
            </Space>
          </Form.Item>
        </Form>
      )}
      
      <div style={{ marginTop: 'var(--spacing-md)' }}>
        <div className="eva-text eva-text--small eva-text--muted">
          Tip: You can find coordinates using Google Maps or GPS apps
        </div>
      </div>
    </div>
  );
};
