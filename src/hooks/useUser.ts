"use client";

import { useCallback, useEffect, useState } from "react";
import { User } from "@prisma/client";
import { useRequest } from "ahooks";
import { useRouter } from "next/navigation";

import { localStore } from "../utils/localStore";
import { API } from "../utils/http";

export const useUser = () => {
  const [user, setUser] = useState<User | null>(null);
  const router = useRouter();
  const { run, loading: logoutLogin } = useRequest(
    async () => {
      const res = API.get("/auth/logout");

      return res;
    },
    {
      manual: true,
      onSuccess() {
        router.replace("/auth/signin");
      },
    },
  );
  const getUserInfo = useCallback(() => {
    const userInfo = localStore.get("user");

    console.log(userInfo, "-------userInfouserInfo-----------");

    if (userInfo) {
      setUser(userInfo);
    }
  }, []);

  const setUserInfo = useCallback((userInfo: User) => {
    setUser(userInfo);
    localStore.set("user", userInfo);
  }, []);

  const logout = useCallback(() => {
    setUser(null);
    localStore.remove("user");
    run();
  }, []);

  useEffect(() => {
    getUserInfo();
  }, []);

  return { user, setUserInfo, logout, logoutLogin };
};
