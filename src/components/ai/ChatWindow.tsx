"use client";
import React, { useRef, useEffect, useMemo, useCallback } from "react";
import { Input, List, Avatar, Button } from "antd";
import { useChat } from "@ai-sdk/react";
import { OpenAIOutlined, UserOutlined } from "@ant-design/icons";
import { UIMessage } from "ai";

import ConentFormat from "./ConentFormat";

const { TextArea } = Input;

interface Message {
  role: "user" | "assistant";
  content: string;
}

const ChatWindow: React.FC = () => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { messages, input, handleInputChange, handleSubmit, status } = useChat();

  const isLoading = useMemo(() => {
    if (status === "streaming" || status === "submitted") {
      return true;
    }

    return false;
  }, [status]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const renderItem = useCallback((msg: UIMessage) => {
    const isMe = msg.role === "user";
    const parts = msg.parts;

    return (
      <List.Item 
        key={msg.id}
        style={{ 
          padding: '8px 0',
          border: 'none'
        }}
      >
        <div style={{
          display: 'flex',
          flexDirection: isMe ? 'row-reverse' : 'row',
          alignItems: 'flex-start',
          gap: '8px',
          width: '100%'
        }}>
          <Avatar 
            icon={isMe ? <UserOutlined /> : <OpenAIOutlined />}
            style={{
              backgroundColor: isMe ? '#0A400C' : '#FE7743',
              flexShrink: 0,
              width: '28px',
              height: '28px'
            }}
            size="small"
          />
          
          <div style={{
            maxWidth: '75%',
            backgroundColor: isMe ? '#e6f7ff' : '#f6ffed',
            borderRadius: '12px',
            padding: '8px 12px',
            fontSize: '13px',
            lineHeight: '1.4',
            wordBreak: 'break-word'
          }}>
            {isMe ? (
              <div>{msg.content}</div>
            ) : (
              parts.map((part, idx) => {
                switch (part.type) {
                  case "text":
                    return <ConentFormat key={idx} content={msg.content} />;
                  case "tool-invocation":
                    switch (part.toolInvocation.state) {
                      case "call":
                        return (
                          <div
                            key={part.toolInvocation.toolCallId}
                            style={{
                              padding: '4px 8px',
                              backgroundColor: 'rgba(0, 0, 0, 0.1)',
                              borderRadius: '6px',
                              fontSize: '12px',
                              color: '#666'
                            }}
                          >
                            Tool call: {part.toolInvocation.toolName} - Calling...
                          </div>
                        );
                      case "result":
                        return <ConentFormat key={idx} content={part.toolInvocation.result} />;
                      default:
                        break;
                    }
                  default:
                    return null;
                }
              })
            )}
          </div>
        </div>
      </List.Item>
    );
  }, []);

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: '12px',
          background: "#fafafa",
        }}
      >
        <List 
          dataSource={messages} 
          renderItem={renderItem}
          size="small"
          split={false}
        />
        <div ref={messagesEndRef} />
      </div>

      <div style={{ 
        padding: '12px',
        borderTop: '1px solid #f0f0f0',
        backgroundColor: 'white',
        display: 'flex',
        gap: '8px'
      }}>
        <TextArea
          disabled={isLoading}
          placeholder="Type a message..."
          rows={1}
          value={input}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          style={{ flex: 1 }}
          size="small"
        />
        <Button 
          loading={isLoading} 
          type="primary" 
          onClick={handleSubmit}
          size="small"
          style={{ minWidth: '50px' }}
        >
          Send
        </Button>
      </div>
    </div>
  );
};

export default ChatWindow;
