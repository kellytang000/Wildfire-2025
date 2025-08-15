import { tool } from "ai";
import { z } from "zod";
const API_KEY = "app-OO3G9Eig8JmsPepI7wFoFlzp";
const WORKFLOW_ID = "f37baca2-f936-49be-a5cf_4ffaedc84262";
const API_URL = "https://api.dify.ai/v1/workflows/run";

export const getFireInfo = tool({
  description: "获取最近的火灾情况",
  parameters: z.object({
    locate: z.string().describe("发生火灾的地名"),
    time: z.string().describe("发生火灾的时间,格式为 2025-07-10T00:00:00"),
  }),
  async execute({ locate, time }) {
    console.log(locate, time, "参数-------");

    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${API_KEY}`,
      },
      body: JSON.stringify({
        workflow_id: WORKFLOW_ID,
        user: "3428538758lxm@gmail.com",
        inputs: {
          locate,
          question: "最近的火情如何",
          time,
        },
      }),
    });

    if (!response.ok) {
      throw new Error(response.statusText);
    }

    const data = await response.json();

    console.log(data, "获取到的火灾信息");

    return data.data.outputs.output;
  },
});
