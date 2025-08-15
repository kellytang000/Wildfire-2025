import { useState, useEffect } from "react";
import { getToken } from "../utils/token";

export interface CurrentUserLocation {
  id: number;
  username: string;
  lat: number | null;
  lng: number | null;
}

// 获取token的辅助函数
const getAuthToken = () => {
  return getToken();
};

// Hook for managing current user location only
export const useCurrentUserLocation = () => {
  const [currentLocation, setCurrentLocation] = useState<CurrentUserLocation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<any>(null);

  // Fetch current user location
  const fetchCurrentLocation = async () => {
    try {
      setLoading(true);
      setError(null);

      const token = getAuthToken();
      if (!token) {
        setCurrentLocation(null);
        return;
      }

      const response = await fetch('/api/user/location', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Cache-Control': 'no-cache',
        },
        cache: 'no-store',
      });

      if (!response.ok) {
        throw new Error('Failed to fetch location');
      }

      const result = await response.json();
      if (result.statusCode === 200) {
        setCurrentLocation(result.data);
      } else {
        throw new Error(result.error || 'Failed to fetch location');
      }
    } catch (error) {
      console.error('Error fetching current location:', error);
      setError(error);
      setCurrentLocation(null);
    } finally {
      setLoading(false);
    }
  };

  // Update current user location
  const updateLocation = async (lat: number, lng: number) => {
    try {
      setLoading(true);
      setError(null);

      const token = getAuthToken();
      if (!token) {
        throw new Error('No authentication token');
      }

      const response = await fetch('/api/user/location', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({ lat, lng }),
      });

      if (!response.ok) {
        throw new Error('Failed to update location');
      }

      const result = await response.json();
      if (result.statusCode === 200) {
        setCurrentLocation(result.data);
        // Immediately fetch the latest location to ensure consistency
        setTimeout(() => {
          fetchCurrentLocation();
        }, 100);
        return result.data;
      } else {
        throw new Error(result.error || 'Failed to update location');
      }
    } catch (error) {
      console.error('Error updating location:', error);
      setError(error);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCurrentLocation();
  }, []);

  return {
    currentLocation,
    loading,
    error,
    updateLocation,
    refresh: fetchCurrentLocation,
  };
};
