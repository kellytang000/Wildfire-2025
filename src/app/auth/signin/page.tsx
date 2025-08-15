"use client";
import { LockOutlined, UserOutlined } from "@ant-design/icons";
import { useRequest } from "ahooks";
import { Button, Form, Input, Select, message } from "antd";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { User } from "@prisma/client";
import Image from "next/image";

import { API } from "@/src/utils/http";
import { useUser } from "@/src/hooks/useUser";

type LoginType = "register" | "login";

const Page = () => {
  const [loginType, setLoginType] = useState<LoginType>("login");
  const [showLoginForm, setShowLoginForm] = useState(false);
  const [form] = Form.useForm();
  const router = useRouter();
  const { setUserInfo } = useUser();

  const { run, loading } = useRequest(
    async (data: Record<string, any>) => {
      const res = API.post<User>("/auth/login", {
        data,
      });

      return res;
    },
    {
      manual: true,
      onSuccess(data, params) {
        setUserInfo(data);
        message.success("Login successful");
        router.replace("/");
      },
    },
  );

  const { run: registerRun, loading: registerLoading } = useRequest(
    async (data: Record<string, any>) => {
      const res = API.post<User>("/auth/register", {
        data,
      });

      return res;
    },
    {
      manual: true,
      onSuccess(data, params) {
        message.success("Registration successful");
        setUserInfo(data);
        router.replace("/");
      },
    },
  );

  const handleSubmit = async (values: any) => {
    const { username, password, confirmPassword, role } = values;

    if (loginType === "login") {
      run({
        username,
        password,
      });
    } else {
      if (password !== confirmPassword) {
        message.error("Passwords do not match");
        return;
      }
      registerRun({
        username,
        password,
        role,
      });
    }
  };

  return (
    <div className="min-h-screen flex">
      {/* Left side - Illustration with fixed height, auto width */}
      <div className="relative flex-shrink-0 h-screen">
        <Image
          src="/background.png"
          alt="Wildfire Emergency Illustration"
          width={0}
          height={0}
          sizes="100vh"
          className="h-full w-auto object-cover"
          priority
        />
      </div>

      {/* Right side - Login Form */}
      <div 
        className={`flex-1 flex items-center px-8 py-12 transition-all duration-500 ${
          showLoginForm ? 'justify-center' : 'justify-start pl-24'
        }`}
        style={{ backgroundColor: '#364532' }}
      >
        <div className={`w-full transition-all duration-500 ${
          showLoginForm ? 'max-w-md' : 'max-w-xl ml-8'
        }`}>
          {!showLoginForm ? (
            /* Initial Landing View */
            <div className="text-left mt-[-60px] animate-fade-in">
              <h1 className="text-6xl font mb-3" style={{ color: '#FAF6E3' }}>Wildfire Emergency</h1>
              <p className="text-xl mb-10" style={{ color: '#FAF6E3', opacity: 0.9 }}>Risk Control & Management System</p>
              <Button
                type="default"
                size="large"
                onClick={() => setShowLoginForm(true)}
                className="px-20 py-4 h-auto text-xl font-medium rounded-full border-2 w-full transition-all duration-200"
                style={{ 
                  backgroundColor: 'transparent', 
                  borderColor: '#FAF6E3',
                  color: '#FAF6E3'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(250, 246, 227, 0.1)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'transparent';
                }}
              >
                Login
              </Button>
            </div>
          ) : (
            /* Login Form View */
            <div className="animate-fade-in">
              {/* Logo and Title */}
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold mb-2" style={{ color: '#FAF6E3' }}>
                  {loginType === "login" ? "Welcome Back" : "Create Account"}
                </h2>
                <p style={{ color: '#FAF6E3', opacity: 0.8 }}>
                  {loginType === "login" 
                    ? "Sign in to access your dashboard" 
                    : "Join us to start managing wildfire risks"
                  }
                </p>
              </div>

              {/* Form */}
              <Form
                form={form}
                layout="vertical"
                onFinish={handleSubmit}
                autoComplete="off"
                size="large"
              >
                <Form.Item
                  label={<span style={{ color: '#FAF6E3' }}>Username</span>}
                  name="username"
                  rules={[
                    { required: true, message: "Please enter your username!" },
                    ...(loginType === "register" ? [{ min: 5, message: "Username must be at least 5 characters long" }] : [])
                  ]}
                >
                  <Input
                    prefix={<UserOutlined className="text-gray-400" />}
                    placeholder="Enter your username"
                    className="rounded-lg border-gray-200 hover:border-blue-400 focus:border-blue-500"
                  />
                </Form.Item>

                <Form.Item
                  label={<span style={{ color: '#FAF6E3' }}>Password</span>}
                  name="password"
                  rules={[
                    { required: true, message: "Please enter your password!" },
                    ...(loginType === "register" ? [{ min: 5, message: "Password must be at least 5 characters long" }] : [])
                  ]}
                >
                  <Input.Password
                    prefix={<LockOutlined className="text-gray-400" />}
                    placeholder="Enter your password"
                    className="rounded-lg border-gray-200 hover:border-blue-400 focus:border-blue-500"
                  />
                </Form.Item>

                {loginType === "register" && (
                  <>
                    <Form.Item
                      label={<span style={{ color: '#FAF6E3' }}>Confirm Password</span>}
                      name="confirmPassword"
                      rules={[
                        { required: true, message: "Please confirm your password!" }
                      ]}
                    >
                      <Input.Password
                        prefix={<LockOutlined className="text-gray-400" />}
                        placeholder="Confirm your password"
                        className="rounded-lg border-gray-200 hover:border-blue-400 focus:border-blue-500"
                      />
                    </Form.Item>

                    <Form.Item
                      label={<span style={{ color: '#FAF6E3' }}>Role</span>}
                      name="role"
                      initialValue="USER"
                      rules={[
                        { required: true, message: "Please select your role!" }
                      ]}
                    >
                      <Select
                        placeholder="Select your role"
                        className="rounded-lg"
                        options={[
                          { label: "User", value: "USER" },
                          { label: "Admin", value: "ADMIN" },
                        ]}
                      />
                    </Form.Item>
                  </>
                )}

                <Form.Item className="mb-6">
                  <Button
                    type="default"
                    htmlType="submit"
                    loading={loading || registerLoading}
                    className="w-full h-12 rounded-lg border-none text-lg font-medium"
                    style={{ 
                      backgroundColor: '#697565',
                      color: '#FAF6E3',
                      border: 'none' 
                    }}
                  >
                    {loginType === "login" ? "Sign In" : "Create Account"}
                  </Button>
                </Form.Item>
              </Form>

              {/* Toggle between login and register */}
              <div className="text-center">
                <span style={{ color: '#FAF6E3', opacity: 0.8 }}>
                  {loginType === "login" ? "Don't have an account? " : "Already have an account? "}
                </span>
                <Button
                  type="link"
                  onClick={() => {
                    setLoginType(loginType === "login" ? "register" : "login");
                    form.resetFields();
                  }}
                  className="font-medium p-0"
                  style={{ color: '#FAF6E3' }}
                >
                  {loginType === "login" ? "Sign up" : "Sign in"}
                </Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Page;
