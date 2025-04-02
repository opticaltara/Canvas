import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { cn } from '../utils/cn';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, className }) => {
  return (
    <div className={cn('markdown-body prose dark:prose-invert max-w-none', className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ node, ...props }) => (
            <h1 {...props} className="text-2xl font-bold mt-6 mb-4" />
          ),
          h2: ({ node, ...props }) => (
            <h2 {...props} className="text-xl font-bold mt-5 mb-3" />
          ),
          h3: ({ node, ...props }) => (
            <h3 {...props} className="text-lg font-bold mt-4 mb-2" />
          ),
          h4: ({ node, ...props }) => (
            <h4 {...props} className="text-base font-bold mt-3 mb-2" />
          ),
          h5: ({ node, ...props }) => (
            <h5 {...props} className="text-sm font-bold mt-3 mb-1" />
          ),
          h6: ({ node, ...props }) => (
            <h6 {...props} className="text-xs font-bold mt-3 mb-1" />
          ),
          p: ({ node, ...props }) => <p {...props} className="my-2" />,
          a: ({ node, ...props }) => (
            <a
              {...props}
              className="text-blue-600 dark:text-blue-400 hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            />
          ),
          ul: ({ node, ...props }) => (
            <ul {...props} className="list-disc pl-5 my-2" />
          ),
          ol: ({ node, ...props }) => (
            <ol {...props} className="list-decimal pl-5 my-2" />
          ),
          li: ({ node, ...props }) => <li {...props} className="my-1" />,
          blockquote: ({ node, ...props }) => (
            <blockquote
              {...props}
              className="border-l-4 border-gray-300 dark:border-gray-700 pl-4 py-1 my-2 italic text-gray-700 dark:text-gray-300"
            />
          ),
          hr: ({ node, ...props }) => (
            <hr {...props} className="my-4 border-gray-300 dark:border-gray-700" />
          ),
          table: ({ node, ...props }) => (
            <div className="overflow-x-auto my-4">
              <table
                {...props}
                className="min-w-full divide-y divide-gray-300 dark:divide-gray-700 border border-gray-300 dark:border-gray-700"
              />
            </div>
          ),
          thead: ({ node, ...props }) => (
            <thead
              {...props}
              className="bg-gray-100 dark:bg-gray-800"
            />
          ),
          tbody: ({ node, ...props }) => (
            <tbody
              {...props}
              className="divide-y divide-gray-200 dark:divide-gray-700"
            />
          ),
          tr: ({ node, ...props }) => (
            <tr
              {...props}
              className="hover:bg-gray-50 dark:hover:bg-gray-800/50"
            />
          ),
          th: ({ node, ...props }) => (
            <th
              {...props}
              className="px-4 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
            />
          ),
          td: ({ node, ...props }) => (
            <td {...props} className="px-4 py-2 whitespace-nowrap" />
          ),
          code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '');
            return !inline && match ? (
              <SyntaxHighlighter
                style={vscDarkPlus}
                language={match[1]}
                PreTag="div"
                className="rounded-md my-4"
                {...props}
              >
                {String(children).replace(/\n$/, '')}
              </SyntaxHighlighter>
            ) : (
              <code
                className={cn(
                  'bg-gray-100 dark:bg-gray-800 rounded px-1 py-0.5 text-sm',
                  className
                )}
                {...props}
              >
                {children}
              </code>
            );
          },
          pre({ node, ...props }) {
            return (
              <pre
                className="bg-gray-100 dark:bg-gray-800 rounded-md overflow-x-auto p-4 my-4"
                {...props}
              />
            );
          },
          strong: ({ node, ...props }) => (
            <strong {...props} className="font-bold" />
          ),
          em: ({ node, ...props }) => <em {...props} className="italic" />,
          img: ({ node, ...props }) => (
            <img
              {...props}
              className="max-w-full h-auto my-4 rounded-md"
              alt={props.alt || 'Image'}
            />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;