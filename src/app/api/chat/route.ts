import { createOpenAI } from "@ai-sdk/openai";
import { streamText } from "ai";
import { createDifyProvider } from "dify-ai-provider";

import { getFireInfo } from "@/src/tool/dify";

const openai = createOpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const difyProvider = createDifyProvider({
  baseURL: "https://cloud.dify.ai/app/f37baca2-f936-49be-a5cf-4ffaedc04262/workflow",
});
const dify = difyProvider("dify-application-id", {
  responseMode: "blocking",
  apiKey: process.env.DIFY_API_KEY,
});

// const dify = difyProvider("f37baca2-f936-49be-a5cf-4ffaedc04262");

// 返回流式数据
export async function POST(request: Request) {
  const { messages } = await request.json();

  // 代理 7890 端口
  const result = streamText({
    // model: deepseek("deepseek-chat"),
    // model: dify,
    model: openai("gpt-4o"),
    // system: "You are a friendly assistant!",
    messages,
    maxSteps: 5,
    tools: { getFireInfo },
  });

  return result.toDataStreamResponse();
}
