"use client";

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Card, Button, Layout } from 'antd';
import { SafetyOutlined, FireOutlined, UserOutlined, SettingOutlined } from '@ant-design/icons';

const { Header, Content } = Layout;

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  const adminMenuItems = [
    {
      href: '/admin/custom-fires',
      label: '自定义火灾管理',
      icon: FireOutlined,
      description: '管理自定义火灾数据'
    },
    {
      href: '/admin/users',
      label: '用户管理',
      icon: UserOutlined,
      description: '管理系统用户'
    },
    {
      href: '/admin/settings',
      label: '系统设置',
      icon: SettingOutlined,
      description: '系统配置管理'
    }
  ];

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#fff', borderBottom: '1px solid #f0f0f0', padding: '0 24px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: '100%' }}>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <SafetyOutlined style={{ fontSize: '24px', color: '#1890ff', marginRight: '8px' }} />
            <h1 style={{ margin: 0, fontSize: '20px', fontWeight: 600 }}>
              管理员面板
            </h1>
          </div>
          <Link href="/">
            <Button>
              返回主页
            </Button>
          </Link>
        </div>
      </Header>

      <Content style={{ padding: '24px', maxWidth: '1200px', margin: '0 auto', width: '100%' }}>
        {pathname === '/admin' ? (
          <div>
            <h2 style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '24px' }}>
              管理员控制台
            </h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px' }}>
              {adminMenuItems.map((item) => {
                const Icon = item.icon;
                return (
                  <Link key={item.href} href={item.href}>
                    <Card
                      hoverable
                      style={{ height: '100%' }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '16px' }}>
                        <Icon style={{ fontSize: '32px', color: '#1890ff', marginRight: '12px' }} />
                        <h3 style={{ margin: 0, fontSize: '18px', fontWeight: 600 }}>
                          {item.label}
                        </h3>
                      </div>
                      <p style={{ margin: 0, color: '#666' }}>
                        {item.description}
                      </p>
                    </Card>
                  </Link>
                );
              })}
            </div>
          </div>
        ) : (
          <div>
            <div style={{ marginBottom: '24px' }}>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {adminMenuItems.map((item) => {
                  const Icon = item.icon;
                  const isActive = pathname === item.href;
                  return (
                    <Link key={item.href} href={item.href}>
                      <Button
                        type={isActive ? "primary" : "default"}
                        icon={<Icon />}
                      >
                        {item.label}
                      </Button>
                    </Link>
                  );
                })}
              </div>
            </div>
            {children}
          </div>
        )}
      </Content>
    </Layout>
  );
}
