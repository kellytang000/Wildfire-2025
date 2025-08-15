import { useRequest } from "ahooks";

import { API } from "../utils/http";
import { IRoutePath } from "../context/map.ctx";

export const useEscape = () => {
  const { runAsync: createEscape, loading } = useRequest(
    (body: { latitude: number; longitude: number }) => {
      return API.post<IRoutePath>("/escape", {
        data: body,
      });
    },
    {
      manual: true,
    },
  );

  return { createEscape, loading };
};
