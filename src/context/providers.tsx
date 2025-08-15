"use client";

import type { ThemeProviderProps } from "next-themes";

import { AntdRegistry } from "@ant-design/nextjs-registry";
import { ConfigProvider } from "antd";
import * as React from "react";
import { useRequest } from "ahooks";
import { User } from "@prisma/client";

import { API } from "../utils/http";
import { useUser } from "../hooks/useUser";

export interface ProvidersProps {
  themeProps?: ThemeProviderProps;
}

export const Providers: React.FC<React.PropsWithChildren<ProvidersProps>> = ({ children, themeProps }) => {
  const { setUserInfo } = useUser();

  useRequest(
    () => {
      return API.get<User>("/auth/check");
    },
    {
      onSuccess: (data) => {
        setUserInfo(data);
      },
    },
  );

  return (
    <AntdRegistry>
      <ConfigProvider>{children}</ConfigProvider>
    </AntdRegistry>
  );
};
