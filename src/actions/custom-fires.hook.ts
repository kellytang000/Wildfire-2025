import { useState, useEffect } from "react";

export interface CustomFirePoint {
  id: number;
  lat: number;
  lng: number;
  properties: {
    OBJECTID: number;
    type: string;
    Shape__Area: number;
    Shape__Length: number;
  };
  geometry: {
    type: "Polygon";
    coordinates: number[][][];
  };
}

// Global refresh trigger
let refreshTrigger = 0;
const refreshCallbacks: Set<() => void> = new Set();

export const triggerGlobalRefresh = () => {
  refreshTrigger++;
  refreshCallbacks.forEach(callback => callback());
};

export const useCustomFires = () => {
  const [customFires, setCustomFires] = useState<CustomFirePoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<any>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const fetchCustomFires = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/custom-fires');

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (result.statusCode === 200 && Array.isArray(result.data)) {
        setCustomFires(result.data);
      } else {
        console.error('Unexpected API response format:', result);
        setCustomFires([]);
      }
    } catch (err) {
      console.error('Error fetching custom fires:', err);
      setError(err);
      setCustomFires([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCustomFires();
  }, []);

  useEffect(() => {
    const handleRefresh = () => {
      setRefreshKey(prev => prev + 1);
      fetchCustomFires();
    };

    refreshCallbacks.add(handleRefresh);
    return () => {
      refreshCallbacks.delete(handleRefresh);
    };
  }, []);

  const createCustomFire = async (fireData: {
    lat: number;
    lng: number;
    type: string;
    geometry: any;
  }) => {
    try {
      const response = await fetch('/api/custom-fires', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(fireData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (result.statusCode === 201) {
        await fetchCustomFires(); // 刷新列表
        triggerGlobalRefresh(); // 触发全局刷新
        return result.data;
      } else {
        throw new Error(result.error || 'Failed to create custom fire');
      }
    } catch (err) {
      console.error('Error creating custom fire:', err);
      throw err;
    }
  };

  const updateCustomFire = async (id: number, fireData: {
    lat: number;
    lng: number;
    type: string;
    geometry: any;
  }) => {
    try {
      const response = await fetch('/api/custom-fires', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ id, ...fireData }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (result.statusCode === 200) {
        await fetchCustomFires(); // 刷新列表
        triggerGlobalRefresh(); // 触发全局刷新
        return result.data;
      } else {
        throw new Error(result.error || 'Failed to update custom fire');
      }
    } catch (err) {
      console.error('Error updating custom fire:', err);
      throw err;
    }
  };

  const deleteCustomFire = async (id: number) => {
    try {
      const response = await fetch(`/api/custom-fires?id=${id}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (result.statusCode === 200) {
        await fetchCustomFires(); // 刷新列表
        triggerGlobalRefresh(); // 触发全局刷新
        return true;
      } else {
        throw new Error(result.error || 'Failed to delete custom fire');
      }
    } catch (err) {
      console.error('Error deleting custom fire:', err);
      throw err;
    }
  };

  return {
    customFires,
    loading,
    error,
    refresh: fetchCustomFires,
    create: createCustomFire,
    update: updateCustomFire,
    delete: deleteCustomFire,
  };
};
