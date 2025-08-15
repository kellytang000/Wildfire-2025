"use client";

import React, { useState, useEffect } from "react";
import {
  Button,
  Form,
  Input,
  InputNumber,
  Select,
  Modal,
  Table,
  message,
  Space,
  Popconfirm,
  Card,
  Tag,
  Tooltip,
} from "antd";
import { PlusOutlined, EditOutlined, DeleteOutlined, HomeOutlined, ReloadOutlined } from "@ant-design/icons";

import { useCreateShelter, useUpdateShelter, useDeleteShelter, useShelters, ShelterData } from "./_hook/shelters.hook";

import { useMapContext } from "@/src/context/map.ctx";

const { Option } = Select;

interface ShelterFormData {
  shelterId: string;
  lat: number;
  lng: number;
  capacity: number;
  hexId: string;
  region: string;
}

const ShelterManager: React.FC = () => {
  const [form] = Form.useForm();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingShelter, setEditingShelter] = useState<ShelterData | null>(null);
  const [selectedRegion, setSelectedRegion] = useState<string>("");

  const { shelters, loading, refresh } = useShelters(selectedRegion || undefined);
  const { createShelter, loading: createLoading } = useCreateShelter();
  const { updateShelter, loading: updateLoading } = useUpdateShelter();
  const { deleteShelter, loading: deleteLoading } = useDeleteShelter();



  // 获取地图上下文中的刷新函数
  const { refreshShelters: refreshMapShelters } = useMapContext();

  // 手动刷新函数
  const handleRefresh = async () => {
    try {
      await Promise.all([refresh(), refreshMapShelters()]);
      message.success("Data refreshed successfully");
    } catch (error) {
      message.error("Refresh failed");
    }
  };

  const handleSubmit = async (values: ShelterFormData) => {
    try {
      let result;

      if (editingShelter) {
        result = await updateShelter({ ...values, id: editingShelter.id });
        message.success("Shelter updated successfully");
      } else {
        result = await createShelter(values);
        message.success("Shelter added successfully");
      }

      // 同时刷新管理界面和地图的数据
      await Promise.all([refresh(), refreshMapShelters()]);

      setIsModalOpen(false);
      setEditingShelter(null);
      form.resetFields();
    } catch (error: any) {
      // 检查是否是唯一约束错误
      if (error?.message?.includes("Unique constraint failed") || error?.message?.includes("shelterId")) {
        message.error(`Shelter ID "${values.shelterId}" already exists, please use a different ID`);
      } else {
        message.error(`Operation failed: ${error?.message || "Unknown error"}`);
      }

      // Try to refresh data even if error occurs, in case data was actually saved
      try {
        await Promise.all([refresh(), refreshMapShelters()]);
      } catch (refreshError) {
        // Silently handle refresh errors
      }
    }
  };

  const handleEdit = (shelter: ShelterData) => {
    setEditingShelter(shelter);
    form.setFieldsValue({
      shelterId: shelter.shelterId,
      lat: shelter.lat,
      lng: shelter.lng,
      capacity: shelter.capacity,
      hexId: shelter.hexId,
      region: shelter.region,
    });
    setIsModalOpen(true);
  };

  const handleDelete = async (id: number, shelterId: string) => {
    try {
      const result = await deleteShelter(id);

      message.success(`Shelter ${shelterId} deleted successfully`);
      await Promise.all([refresh(), refreshMapShelters()]);
    } catch (error: any) {
      message.error(`Delete failed: ${error?.message || "Unknown error"}`);

      // Try to refresh data even if error occurs
      try {
        await Promise.all([refresh(), refreshMapShelters()]);
      } catch (refreshError) {
        // Silently handle refresh errors
      }
    }
  };

  const generateUniqueShelterID = (region: string) => {
    const timestamp = Date.now();
    const prefix = region === "squamish" ? "squa" : "la";

    return `${prefix}_${timestamp}`;
  };

  const handleAdd = () => {
    setEditingShelter(null);
    form.resetFields();
    // 预设一个建议的shelter ID
    const suggestedId = generateUniqueShelterID("squamish");

    form.setFieldsValue({
      shelterId: suggestedId,
      region: "squamish",
    });
    setIsModalOpen(true);
  };

  const columns = [
    {
      title: "Shelter ID",
      dataIndex: "shelterId",
      key: "shelterId",
      width: 150,
      render: (text: string) => (
        <Tag color="blue" icon={<HomeOutlined />}>
          {text}
        </Tag>
      ),
    },
    {
      title: "Region",
      dataIndex: "region",
      key: "region",
      width: 120,
      render: (region: string) => (
        <Tag color={region === "squamish" ? "green" : "orange"}>
          {region === "squamish" ? "Squamish" : "Los Angeles"}
        </Tag>
      ),
    },
    {
      title: "Coordinates",
      key: "coordinates",
      width: 200,
      render: (_: any, record: ShelterData) => (
        <div className="text-sm">
          <div>Lat: {record.lat.toFixed(6)}</div>
          <div>Lng: {record.lng.toFixed(6)}</div>
        </div>
      ),
    },
    {
      title: "Capacity",
      dataIndex: "capacity",
      key: "capacity",
      width: 100,
      render: (capacity: number) => <Tag color="purple">{capacity.toLocaleString()}</Tag>,
    },
    {
      title: "Hex ID",
      dataIndex: "hexId",
      key: "hexId",
      width: 150,
      ellipsis: true,
      render: (hexId: string) => (
        <Tooltip title={hexId}>
          <span className="text-gray-600">{hexId}</span>
        </Tooltip>
      ),
    },
    {
      title: "Created At",
      dataIndex: "createdAt",
      key: "createdAt",
      width: 150,
      render: (date: string) => new Date(date).toLocaleDateString("en-US"),
    },
    {
      title: "Actions",
      key: "actions",
      width: 150,
      render: (_: any, record: ShelterData) => (
        <Space>
          <Tooltip title="Edit">
            <Button icon={<EditOutlined />} size="small" type="link" onClick={() => handleEdit(record)} />
          </Tooltip>
          <Popconfirm
            cancelText="No"
            description={`Are you sure you want to delete ${record.shelterId}?`}
            okText="Yes"
            title="Confirm Delete"
            onConfirm={() => handleDelete(record.id, record.shelterId)}
          >
            <Tooltip title="Delete">
              <Button danger icon={<DeleteOutlined />} loading={deleteLoading} size="small" type="link" />
            </Tooltip>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  const regionStats = {
    total: shelters.length,
    squamish: shelters.filter((s) => s.region === "squamish").length,
    la: shelters.filter((s) => s.region === "la").length,
    totalCapacity: shelters.reduce((sum, s) => sum + s.capacity, 0),
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <Card className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-800 mb-2">
              <HomeOutlined className="mr-2" />
              Shelter Management System
            </h1>
            <p className="text-gray-600">
              Manage emergency shelter information, support add, edit and delete operations
            </p>
          </div>
          <Space>
            <Button icon={<ReloadOutlined />} loading={loading} onClick={handleRefresh}>
              Refresh
            </Button>
            <Button icon={<PlusOutlined />} size="large" type="primary" onClick={handleAdd}>
              Add Shelter
            </Button>
          </Space>
        </div>

        {/* Statistics */}
        <div className="grid grid-cols-4 gap-4 mb-4">
          <div className="bg-blue-50 p-4 rounded-lg text-center">
            <div className="text-2xl font-bold text-blue-600">{regionStats.total}</div>
            <div className="text-sm text-gray-600">Total</div>
          </div>
          <div className="bg-green-50 p-4 rounded-lg text-center">
            <div className="text-2xl font-bold text-green-600">{regionStats.squamish}</div>
            <div className="text-sm text-gray-600">Squamish</div>
          </div>
          <div className="bg-orange-50 p-4 rounded-lg text-center">
            <div className="text-2xl font-bold text-orange-600">{regionStats.la}</div>
            <div className="text-sm text-gray-600">Los Angeles</div>
          </div>
          <div className="bg-purple-50 p-4 rounded-lg text-center">
            <div className="text-2xl font-bold text-purple-600">{regionStats.totalCapacity.toLocaleString()}</div>
            <div className="text-sm text-gray-600">Total Capacity</div>
          </div>
        </div>

        {/* Filter */}
        <div className="flex gap-4 mb-4">
          <Select
            allowClear
            placeholder="Select region to filter"
            style={{ width: 200 }}
            value={selectedRegion}
            onChange={setSelectedRegion}
          >
            <Option value="squamish">Squamish</Option>
            <Option value="la">Los Angeles</Option>
          </Select>
        </div>
      </Card>

      {/* Table */}
      <Card>
        <Table
          columns={columns}
          dataSource={shelters}
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `${range[0]}-${range[1]} of ${total} items`,
          }}
          rowKey="id"
          scroll={{ x: 1000 }}
        />
      </Card>

      {/* Add/Edit Modal */}
      <Modal
        footer={null}
        open={isModalOpen}
        title={
          <div className="flex items-center">
            <HomeOutlined className="mr-2" />
            {editingShelter ? "Edit Shelter" : "Add Shelter"}
          </div>
        }
        width={600}
        onCancel={() => {
          setIsModalOpen(false);
          setEditingShelter(null);
          form.resetFields();
        }}
      >
        <Form className="mt-4" form={form} layout="vertical" onFinish={handleSubmit}>
          <Form.Item
            extra={
              <div className="text-xs text-gray-500 mt-1">
                <div>Existing IDs: {shelters.map((s) => s.shelterId).join(", ")}</div>
                <div>Suggested format: squa_[timestamp] or la_[timestamp]</div>
              </div>
            }
            label="Shelter ID"
            name="shelterId"
            rules={[
              { required: true, message: "Please enter Shelter ID" },
              { pattern: /^[a-zA-Z0-9_-]+$/, message: "Only letters, numbers, underscores and hyphens are allowed" },
            ]}
          >
            <Input placeholder="System will auto-generate unique ID" />
          </Form.Item>

          <Form.Item label="Region" name="region" rules={[{ required: true, message: "Please select a region" }]}>
            <Select
              placeholder="Select region"
              onChange={(region) => {
                // Auto-generate new shelter ID when region changes (if not in edit mode)
                if (!editingShelter) {
                  const newId = generateUniqueShelterID(region);

                  form.setFieldsValue({ shelterId: newId });
                }
              }}
            >
              <Option value="squamish">Squamish (British Columbia, Canada)</Option>
              <Option value="la">Los Angeles (California, USA)</Option>
            </Select>
          </Form.Item>

          <div className="grid grid-cols-2 gap-4">
            <Form.Item
              label="Latitude"
              name="lat"
              rules={[
                { required: true, message: "Please enter latitude" },
                { type: "number", min: -90, max: 90, message: "Latitude range: -90 to 90" },
              ]}
            >
              <InputNumber placeholder="e.g.: 49.7500" precision={6} step={0.000001} style={{ width: "100%" }} />
            </Form.Item>

            <Form.Item
              label="Longitude"
              name="lng"
              rules={[
                { required: true, message: "Please enter longitude" },
                { type: "number", min: -180, max: 180, message: "Longitude range: -180 to 180" },
              ]}
            >
              <InputNumber placeholder="e.g.: -123.1200" precision={6} step={0.000001} style={{ width: "100%" }} />
            </Form.Item>
          </div>

          <Form.Item
            label="Capacity"
            name="capacity"
            rules={[
              { required: true, message: "Please enter capacity" },
              { type: "number", min: 1, message: "Capacity must be greater than 0" },
            ]}
          >
            <InputNumber
              formatter={(value) => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
              min={1}
              placeholder="e.g.: 500"
              style={{ width: "100%" }}
            />
          </Form.Item>

          <Form.Item label="Hex ID" name="hexId" rules={[{ required: true, message: "Please enter Hex ID" }]}>
            <Input placeholder="Hexagonal grid ID" />
          </Form.Item>

          <div className="flex justify-end gap-2 mt-6">
            <Button onClick={() => setIsModalOpen(false)}>Cancel</Button>
            <Button htmlType="submit" loading={createLoading || updateLoading} type="primary">
              {editingShelter ? "Update" : "Add"}
            </Button>
          </div>
        </Form>
      </Modal>
    </div>
  );
};

export default ShelterManager;
