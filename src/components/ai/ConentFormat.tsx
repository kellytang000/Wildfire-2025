import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import oneDark from "react-syntax-highlighter/dist/esm/styles/prism/one-dark";
import { FC, PropsWithChildren } from "react";

interface IProps {
  content: string;
}

const ConentFormat: FC<PropsWithChildren<IProps>> = ({ content }) => {
  return (
    <ReactMarkdown
      components={{
        code({ node, inline, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || "");

          return !inline && match ? (
            <SyntaxHighlighter PreTag="div" language={match[1]} style={oneDark} {...props}>
              {String(children).replace(/\n$/, "")}
            </SyntaxHighlighter>
          ) : (
            <code className={className} {...props}>
              {children}
            </code>
          );
        },
      }}
      remarkPlugins={[remarkGfm]} // 支持表格、待办等 GFM 特性
    >
      {content}
    </ReactMarkdown>
  );
};

export default ConentFormat;
