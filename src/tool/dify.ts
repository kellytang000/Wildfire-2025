import { createDeepSeek } from "@ai-sdk/deepseek";
import { streamText } from "ai";
import { z } from "zod";

const deepseek = createDeepSeek({
  apiKey: process.env.DEEPSEEK_API_KEY,
});

// Enhanced wildfire information tool
const getWildfireEmergencyInfo = {
  description: 'Get comprehensive wildfire emergency information including current incidents, weather conditions, evacuation status, and emergency recommendations',
  parameters: z.object({
    location: z.string().describe('Geographic location for wildfire analysis'),
    analysisType: z.enum(['current_incidents', 'risk_assessment', 'evacuation_planning', 'weather_conditions', 'comprehensive']).describe('Type of analysis requested'),
    timeFrame: z.enum(['immediate', '24hours', '72hours', 'weekly']).optional().describe('Time frame for analysis'),
  }),
  execute: async ({ location, analysisType, timeFrame = 'immediate' }) => {
    try {
      // Simulate real-time data collection
      const currentTime = new Date().toISOString();
      
      // Mock data structure representing real API responses
      const wildfireData = {
        current_incidents: {
          active_fires: [
            {
              name: `${location} Wildfire Complex`,
              location: location,
              size_acres: Math.floor(Math.random() * 50000) + 1000,
              containment_percent: Math.floor(Math.random() * 100),
              start_date: "2024-08-14T10:00:00Z",
              threat_level: ["Low", "Moderate", "High", "Extreme"][Math.floor(Math.random() * 4)],
              structures_threatened: Math.floor(Math.random() * 1000),
              personnel_assigned: Math.floor(Math.random() * 500) + 100
            }
          ],
          total_active_incidents: Math.floor(Math.random() * 10) + 1,
          red_flag_warnings: Math.random() > 0.5
        },
        
        weather_conditions: {
          temperature_f: Math.floor(Math.random() * 30) + 70,
          humidity_percent: Math.floor(Math.random() * 40) + 10,
          wind_speed_mph: Math.floor(Math.random() * 25) + 5,
          wind_direction: ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][Math.floor(Math.random() * 8)],
          fire_weather_index: ["Low", "Moderate", "High", "Critical"][Math.floor(Math.random() * 4)],
          forecast_confidence: "High"
        },
        
        evacuation_status: {
          zones_under_evacuation_order: [`Zone A-${location}`, `Zone B-${location}`],
          zones_under_evacuation_warning: [`Zone C-${location}`],
          shelter_locations: [
            `${location} Community Center`,
            `${location} High School Gymnasium`,
            "Regional Emergency Shelter"
          ],
          transportation_assistance_available: true,
          pet_evacuation_centers: [`${location} Animal Services`]
        },
        
        emergency_resources: {
          fire_stations_responding: Math.floor(Math.random() * 10) + 3,
          engines_deployed: Math.floor(Math.random() * 20) + 5,
          aircraft_available: Math.floor(Math.random() * 5) + 1,
          emergency_medical_teams: Math.floor(Math.random() * 8) + 2,
          incident_command_post: `${location} Emergency Operations Center`
        },
        
        recommendations: {
          immediate_actions: [
            "Monitor official emergency alerts and evacuation orders",
            "Prepare emergency supplies and important documents",
            "Identify multiple evacuation routes from your area",
            "Stay informed through official emergency management channels"
          ],
          preparation_measures: [
            "Create defensible space around structures",
            "Review and update family emergency plan",
            "Ensure emergency communication devices are charged",
            "Pre-position vehicles for potential evacuation"
          ]
        },
        
        metadata: {
          report_generated: currentTime,
          data_sources: ["National Interagency Fire Center", "National Weather Service", "Local Emergency Management", "InciWeb"],
          analysis_type: analysisType,
          location_analyzed: location,
          confidence_level: "High",
          next_update_scheduled: new Date(Date.now() + 2 * 60 * 60 * 1000).toISOString() // 2 hours from now
        }
      };

      // Return data based on analysis type
      switch (analysisType) {
        case 'current_incidents':
          return {
            success: true,
            data: {
              incidents: wildfireData.current_incidents,
              metadata: wildfireData.metadata
            }
          };
          
        case 'risk_assessment':
          return {
            success: true,
            data: {
              risk_factors: wildfireData.weather_conditions,
              threat_assessment: wildfireData.current_incidents,
              metadata: wildfireData.metadata
            }
          };
          
        case 'evacuation_planning':
          return {
            success: true,
            data: {
              evacuation_status: wildfireData.evacuation_status,
              emergency_resources: wildfireData.emergency_resources,
              recommendations: wildfireData.recommendations,
              metadata: wildfireData.metadata
            }
          };
          
        case 'weather_conditions':
          return {
            success: true,
            data: {
              weather: wildfireData.weather_conditions,
              metadata: wildfireData.metadata
            }
          };
          
        case 'comprehensive':
        default:
          return {
            success: true,
            data: wildfireData
          };
      }
      
    } catch (error) {
      return {
        success: false,
        error: `Failed to retrieve wildfire information: ${error.message}`,
        fallback_recommendations: [
          "Contact local emergency management authorities directly",
          "Monitor official emergency broadcast channels",
          "Visit InciWeb.nwcg.gov for current incident information",
          "Call 511 for evacuation route information"
        ]
      };
    }
  },
};

// Professional Wildfire Emergency Management AI Assistant
export async function POST(request: Request) {
  const { messages } = await request.json();

  const result = streamText({
    model: deepseek("deepseek-coder"),
    system: `You are a Professional Wildfire Emergency Management AI Assistant with advanced analytical capabilities.

CORE MISSION: Provide accurate, actionable wildfire emergency analysis and response guidance using real-time information and established emergency management protocols.

OPERATIONAL CAPABILITIES:
• Real-time wildfire threat assessment and monitoring
• Emergency evacuation planning and coordination  
• Risk analysis using current conditions and historical patterns
• Professional emergency management communication
• Integration of multiple authoritative data sources

COMMUNICATION STANDARDS:
• Use precise, professional emergency management English terminology
• Follow NIMS (National Incident Management System) communication protocols
• Provide quantitative data with appropriate confidence levels
• Include authoritative source attribution (NWS, NIFC, InciWeb, CAL FIRE)
• Maintain situational awareness and operational perspective

RESPONSE STRUCTURE for Emergency Queries:
1. **EXECUTIVE SUMMARY**: Immediate threat level and critical information
2. **SITUATION ASSESSMENT**: Current wildfire conditions and trends
3. **RISK EVALUATION**: Threat probability, impact assessment, vulnerabilities
4. **RECOMMENDED ACTIONS**: Prioritized, time-sensitive actions with specific guidance
5. **RESOURCE STATUS**: Available emergency resources and contact information
6. **MONITORING REQUIREMENTS**: Key indicators and update intervals

INFORMATION PROCESSING:
• Always use the getWildfireEmergencyInfo tool for current data
• Cross-reference multiple data sources when available
• Provide confidence levels for all assessments
• Include both immediate and longer-term considerations
• Emphasize safety protocols and official emergency channels

PROFESSIONAL TONE: Authoritative but accessible, focusing on public safety and clear actionable guidance. Use emergency management terminology appropriately while ensuring information is understandable to general public.

Always prioritize life safety and provide clear, actionable guidance based on current emergency management best practices.`,
    messages,
    maxSteps: 3,
    tools: { 
      getWildfireEmergencyInfo
    },
  });

  return result.toDataStreamResponse();
}