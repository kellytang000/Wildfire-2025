import { useRequest } from "ahooks";
import { useState, useEffect } from "react";

import { API } from "@/src/utils/http";

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
  const [shelters, setShelters] = useState<ShelterData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<any>(null);

  const fetchShelters = async () => {
    try {
      setLoading(true);
      setError(null);

      const params = region ? `?region=${region}` : "";
      const response = await fetch(`/api/shelters${params}`);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (result.statusCode === 200 && Array.isArray(result.data)) {
        setShelters(result.data);
      } else {
        console.error('Unexpected API response format:', result);
        setShelters([]);
      }
    } catch (err) {
      console.error('Error fetching shelters:', err);
      setError(err);
      setShelters([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchShelters();
  }, [region]);

  return {
    shelters,
    loading,
    error,
    refresh: fetchShelters,
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
