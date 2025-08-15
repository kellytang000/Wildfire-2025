import { useRequest } from "ahooks";

import { API } from "../utils/http";

export interface ShelterData {
  id: number;
  shelterId: string;
  lat: number;
  lng: number;
  capacity: number;
  hexId: string;
  region: string;
  createdAt: Date;
  updatedAt: Date;
}

export const useShelters = (region?: string) => {
  const { data, loading, error, refresh } = useRequest(
    () => {
      const params = region ? `?region=${region}` : "";

      return API.get<ShelterData[]>(`/shelters${params}`);
    },
    {
      refreshDeps: [region], // 当region变化时自动刷新
    },
  );

  return {
    shelters: data || [],
    loading,
    error,
    refresh,
  };
};

export const useCreateShelter = () => {
  const { runAsync: createShelter, loading } = useRequest(
    (shelterData: Omit<ShelterData, "id" | "createdAt" | "updatedAt">) => {
      return API.post<ShelterData>("/shelters", {
        data: shelterData,
      });
    },
    {
      manual: true,
    },
  );

  return { createShelter, loading };
};

export const useUpdateShelter = () => {
  const { runAsync: updateShelter, loading } = useRequest(
    (shelterData: Partial<ShelterData> & { id: number }) => {
      return API.put<ShelterData>("/shelters", {
        data: shelterData,
      });
    },
    {
      manual: true,
    },
  );

  return { updateShelter, loading };
};

export const useDeleteShelter = () => {
  const { runAsync: deleteShelter, loading } = useRequest(
    (id: number) => {
      return API.delete<ShelterData>(`/shelters?id=${id}`);
    },
    {
      manual: true,
    },
  );

  return { deleteShelter, loading };
};
