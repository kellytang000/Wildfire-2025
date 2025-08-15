"use client";

import React, { useState } from 'react';
import { Button, Card, Table, Space, Popconfirm, message, Modal, Form, Input, InputNumber, Select, Tabs, Row, Col, Typography } from 'antd';
import { EditOutlined, DeleteOutlined, PlusOutlined, EnvironmentOutlined, MinusCircleOutlined } from '@ant-design/icons';
import { useCustomFires, CustomFirePoint } from '@/src/actions/custom-fires.hook';

const { Title, Text } = Typography;

export default function AdminCustomFiresPage() {
  const { customFires, loading, refresh, create: createCustomFire, delete: deleteCustomFire } = useCustomFires();
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [activeTab, setActiveTab] = useState('1');
  const [form] = Form.useForm();

  // Delete fire
  const handleDelete = async (id: number) => {
    try {
      await deleteCustomFire(id);
      message.success('Fire deleted successfully');
    } catch (error) {
      console.error('Error deleting fire:', error);
      message.error('Failed to delete fire');
    }
  };

  // Create new fire
  const handleCreate = async (values: any) => {
    try {
      let geometry;
      let centerLat, centerLng;

      if (activeTab === '1') {
        // Simple coordinates mode
        centerLat = values.lat;
        centerLng = values.lng;
        geometry = {
          type: "Polygon",
          coordinates: [[
            [values.lng - 0.001, values.lat - 0.001],
            [values.lng + 0.001, values.lat - 0.001],
            [values.lng + 0.001, values.lat + 0.001],
            [values.lng - 0.001, values.lat + 0.001],
            [values.lng - 0.001, values.lat - 0.001]
          ]]
        };
      } else {
        // Polygon coordinates mode
        const coordinates = values.coordinates;
        if (!coordinates || coordinates.length < 3) {
          message.error('Please add at least 3 coordinate points');
          return;
        }

        // Convert coordinates to proper format and close the polygon
        const polygonCoords = coordinates.map((coord: any) => [coord.lng, coord.lat]);
        // Close the polygon by adding the first point at the end if not already closed
        if (polygonCoords.length > 0 &&
            (polygonCoords[0][0] !== polygonCoords[polygonCoords.length - 1][0] ||
             polygonCoords[0][1] !== polygonCoords[polygonCoords.length - 1][1])) {
          polygonCoords.push(polygonCoords[0]);
        }

        geometry = {
          type: "Polygon",
          coordinates: [polygonCoords]
        };

        // Calculate center point
        let latSum = 0;
        let lngSum = 0;
        for (const coord of coordinates) {
          lngSum += coord.lng;
          latSum += coord.lat;
        }
        centerLat = latSum / coordinates.length;
        centerLng = lngSum / coordinates.length;
      }

      const fireData = {
        lat: centerLat,
        lng: centerLng,
        type: values.type,
        geometry: geometry
      };

      await createCustomFire(fireData);
      message.success('Fire created successfully');
      setIsModalVisible(false);
      form.resetFields();
    } catch (error) {
      console.error('Error creating fire:', error);
      message.error('Failed to create fire');
    }
  };



  const columns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 80,
    },
    {
      title: 'Type',
      dataIndex: ['properties', 'type'],
      key: 'type',
      width: 150,
    },
    {
      title: 'Latitude',
      dataIndex: 'lat',
      key: 'lat',
      width: 120,
      render: (lat: number) => lat.toFixed(6),
    },
    {
      title: 'Longitude',
      dataIndex: 'lng',
      key: 'lng',
      width: 120,
      render: (lng: number) => lng.toFixed(6),
    },
    {
      title: 'Area',
      dataIndex: ['properties', 'Shape__Area'],
      key: 'area',
      width: 120,
      render: (area: number) => area.toFixed(2),
    },
    {
      title: 'Perimeter',
      dataIndex: ['properties', 'Shape__Length'],
      key: 'length',
      width: 120,
      render: (length: number) => length.toFixed(2),
    },
    {
      title: 'Actions',
      key: 'action',
      width: 100,
      render: (_: any, record: CustomFirePoint) => (
        <Popconfirm
          title="Are you sure you want to delete this custom fire?"
          onConfirm={() => handleDelete(record.id)}
          okText="Yes"
          cancelText="No"
        >
          <Button
            type="link"
            danger
            icon={<DeleteOutlined />}
          >
            Delete
          </Button>
        </Popconfirm>
      ),
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h1 style={{ fontSize: '24px', fontWeight: 'bold', margin: 0 }}>Custom Fire Management</h1>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setIsModalVisible(true)}
        >
          Add New Fire
        </Button>
      </div>

      <Card>
        <Table
          columns={columns}
          dataSource={customFires}
          loading={loading}
          rowKey="id"
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) =>
              `${range[0]}-${range[1]} of ${total} items`,
          }}
          scroll={{ x: 1000 }}
        />
      </Card>

      <Modal
        title="Add New Custom Fire"
        open={isModalVisible}
        onCancel={() => {
          setIsModalVisible(false);
          form.resetFields();
        }}
        footer={null}
        width={900}
      >
        <Tabs
          activeKey={activeTab}
          onChange={setActiveTab}
          items={[
            {
              key: '1',
              label: (
                <span>
                  <EnvironmentOutlined />
                  Simple Coordinates
                </span>
              ),
              children: (
                <Form
                  form={form}
                  layout="vertical"
                  onFinish={handleCreate}
                  style={{ marginTop: '20px' }}
                >
                  <Form.Item
                    label="Fire Type"
                    name="type"
                    rules={[{ required: true, message: 'Please select fire type' }]}
                  >
                    <Select placeholder="Select fire type">
                      <Select.Option value="Heat Perimeter">Heat Perimeter</Select.Option>
                      <Select.Option value="Fire Perimeter">Fire Perimeter</Select.Option>
                      <Select.Option value="Burn Area">Burn Area</Select.Option>
                    </Select>
                  </Form.Item>

                  <Form.Item
                    label="Latitude"
                    name="lat"
                    rules={[
                      { required: true, message: 'Please enter latitude' },
                      { type: 'number', min: -90, max: 90, message: 'Latitude must be between -90 and 90' }
                    ]}
                  >
                    <InputNumber
                      style={{ width: '100%' }}
                      placeholder="Enter latitude (e.g., 34.0522)"
                      step={0.000001}
                      precision={6}
                    />
                  </Form.Item>

                  <Form.Item
                    label="Longitude"
                    name="lng"
                    rules={[
                      { required: true, message: 'Please enter longitude' },
                      { type: 'number', min: -180, max: 180, message: 'Longitude must be between -180 and 180' }
                    ]}
                  >
                    <InputNumber
                      style={{ width: '100%' }}
                      placeholder="Enter longitude (e.g., -118.2437)"
                      step={0.000001}
                      precision={6}
                    />
                  </Form.Item>

                  <Form.Item style={{ marginBottom: 0, textAlign: 'right' }}>
                    <Space>
                      <Button onClick={() => {
                        setIsModalVisible(false);
                        form.resetFields();
                      }}>
                        Cancel
                      </Button>
                      <Button type="primary" htmlType="submit">
                        Create Fire
                      </Button>
                    </Space>
                  </Form.Item>
                </Form>
              ),
            },
            {
              key: '2',
              label: (
                <span>
                  <EditOutlined />
                  Polygon Coordinates
                </span>
              ),
              children: (
                <Form
                  form={form}
                  layout="vertical"
                  onFinish={handleCreate}
                  initialValues={{
                    coordinates: [
                      { lat: 34.0522, lng: -118.2437 },
                      { lat: 34.0532, lng: -118.2427 },
                      { lat: 34.0512, lng: -118.2427 }
                    ]
                  }}
                >
                  <Form.Item
                    label="Fire Type"
                    name="type"
                    rules={[{ required: true, message: 'Please select fire type' }]}
                  >
                    <Select placeholder="Select fire type">
                      <Select.Option value="Heat Perimeter">Heat Perimeter</Select.Option>
                      <Select.Option value="Fire Perimeter">Fire Perimeter</Select.Option>
                      <Select.Option value="Burn Area">Burn Area</Select.Option>
                    </Select>
                  </Form.Item>

                  <div style={{ marginBottom: '16px' }}>
                    <Title level={5}>Polygon Coordinates</Title>
                    <Text type="secondary">
                      Add at least 3 coordinate points to form a polygon. The polygon will be automatically closed.
                    </Text>
                  </div>

                  <Form.List name="coordinates">
                    {(fields, { add, remove }) => (
                      <>
                        {fields.map(({ key, name, ...restField }) => (
                          <Card
                            key={key}
                            size="small"
                            style={{ marginBottom: 8 }}
                            title={`Point ${name + 1}`}
                            extra={
                              fields.length > 3 ? (
                                <Button
                                  type="link"
                                  danger
                                  icon={<MinusCircleOutlined />}
                                  onClick={() => remove(name)}
                                >
                                  Remove
                                </Button>
                              ) : null
                            }
                          >
                            <Row gutter={16}>
                              <Col span={12}>
                                <Form.Item
                                  {...restField}
                                  name={[name, 'lat']}
                                  label="Latitude"
                                  rules={[
                                    { required: true, message: 'Required' },
                                    { type: 'number', min: -90, max: 90, message: 'Invalid latitude' }
                                  ]}
                                >
                                  <InputNumber
                                    style={{ width: '100%' }}
                                    placeholder="34.0522"
                                    step={0.000001}
                                    precision={6}
                                  />
                                </Form.Item>
                              </Col>
                              <Col span={12}>
                                <Form.Item
                                  {...restField}
                                  name={[name, 'lng']}
                                  label="Longitude"
                                  rules={[
                                    { required: true, message: 'Required' },
                                    { type: 'number', min: -180, max: 180, message: 'Invalid longitude' }
                                  ]}
                                >
                                  <InputNumber
                                    style={{ width: '100%' }}
                                    placeholder="-118.2437"
                                    step={0.000001}
                                    precision={6}
                                  />
                                </Form.Item>
                              </Col>
                            </Row>
                          </Card>
                        ))}
                        <Form.Item>
                          <Button
                            type="dashed"
                            onClick={() => add()}
                            block
                            icon={<PlusOutlined />}
                          >
                            Add Coordinate Point
                          </Button>
                        </Form.Item>
                      </>
                    )}
                  </Form.List>

                  <Form.Item style={{ marginBottom: 0, textAlign: 'right' }}>
                    <Space>
                      <Button onClick={() => {
                        setIsModalVisible(false);
                        form.resetFields();
                      }}>
                        Cancel
                      </Button>
                      <Button type="primary" htmlType="submit">
                        Create Fire
                      </Button>
                    </Space>
                  </Form.Item>
                </Form>
              ),
            },
          ]}
        />
      </Modal>
    </div>
  );
}
