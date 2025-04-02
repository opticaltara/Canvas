import { AnthropicSonnet37Tools } from '@anthropic-ai/sdk';
import { StreamingTextResponse } from 'ai';
import { NextRequest, NextResponse } from 'next/server';
import { handleToolCall } from '@/app/lib/ai-agent';
import { sqlQueryTool, logQueryTool, metricQueryTool, s3QueryTool, dataSourceContextTool } from '@/app/lib/tools';

export const runtime = 'edge';
export const maxDuration = 300; // 5 minutes

export async function POST(req: NextRequest) {
  try {
    const { messages, query, requestType, connectionId } = await req.json();
    
    // Validate API key
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      return NextResponse.json(
        { error: 'Anthropic API key is required' },
        { status: 500 }
      );
    }
    
    const anthropic = new AnthropicSonnet37Tools({
      apiKey,
    });
    
    // Handle different request types
    if (requestType === 'investigation_plan') {
      // Basic messages array for investigation plan
      const response = anthropic.messages.stream({
        model: 'claude-3-sonnet-20240229',
        system: `
          You are an expert at software engineering investigation. Your task is to create a detailed plan
          to investigate and solve problems by breaking them down into specific steps.
          
          Create an investigation plan with ordered steps. Each step should:
          1. Have a clear purpose in the investigation
          2. Specify what type of cell to create (sql, log, metric, python, markdown)
          3. Include the actual query/code/content for that cell
          4. List any dependencies on previous steps
          
          Output each step one at a time as a JSON object with these fields:
          - step_id: number (starting from 1)
          - description: string (purpose of this step)
          - cell_type: string (one of: "markdown", "python", "sql", "log", "metric", "s3")
          - content: string (the actual code/query/content for the cell)
          - depends_on: number[] (array of step_ids this step depends on)
          
          Use the available tools to gather information about data sources when needed.
        `,
        max_tokens: 4000,
        messages: [{ role: 'user', content: query }],
        tools: [sqlQueryTool, logQueryTool, metricQueryTool, s3QueryTool, dataSourceContextTool],
      });
      
      // Handle tool calls in the stream
      const responseWithTools = response.onToolCall(async (toolCall) => {
        const result = await handleToolCall(toolCall);
        return { content: JSON.stringify(result) };
      });
      
      // Return a streaming response
      return new StreamingTextResponse(responseWithTools);
      
    } else if (requestType === 'structured_output') {
      // Generate structured output using Anthropic's structured_output feature
      const response = await anthropic.messages.create({
        model: 'claude-3-sonnet-20240229',
        system: `
          You are an expert at software engineering investigation. Create a structured plan to investigate the user's query.
          Think step by step and be thorough in your approach.
        `,
        max_tokens: 4000,
        messages: [{ role: 'user', content: query }],
        tools: [sqlQueryTool, logQueryTool, metricQueryTool, s3QueryTool, dataSourceContextTool],
        tool_choice: 'auto',
        structured_output: {
          schema: {
            type: 'object',
            properties: {
              steps: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    step_id: { type: 'number' },
                    description: { type: 'string' },
                    cell_type: { 
                      type: 'string',
                      enum: ['markdown', 'python', 'sql', 'log', 'metric', 's3']
                    },
                    content: { type: 'string' },
                    depends_on: { 
                      type: 'array',
                      items: { type: 'number' }
                    }
                  },
                  required: ['step_id', 'description', 'cell_type', 'content', 'depends_on']
                }
              },
              thinking: { type: 'string' }
            },
            required: ['steps']
          }
        }
      });
      
      // Extract structured output from the response
      const content = response.content[0];
      if (content.type === 'structured_output') {
        return NextResponse.json({ plan: content.structured_output });
      } else {
        throw new Error('Expected structured output but received text');
      }
      
    } else if (requestType === 'query_generation') {
      // Handle cell-specific query generation
      const cellType = messages[0]?.content || 'markdown';
      let systemPrompt = '';
      
      // Set system prompt based on cell type
      switch (cellType) {
        case 'sql':
          systemPrompt = 'You are an expert SQL query writer. Generate a SQL query that addresses the request. Return ONLY the SQL query with no explanations.';
          break;
        case 'log':
          systemPrompt = 'You are an expert in log analysis. Generate a log query that addresses the request. Return ONLY the log query with no explanations.';
          break;
        case 'metric':
          systemPrompt = 'You are an expert in metric analysis. Generate a PromQL query that addresses the request. Return ONLY the PromQL query with no explanations.';
          break;
        case 'python':
          systemPrompt = 'You are an expert Python programmer. Generate Python code that addresses the request. Return ONLY the Python code with no explanations.';
          break;
        case 'markdown':
          systemPrompt = 'You are an expert at technical documentation. Create clear markdown to address the request. Return ONLY the markdown with no meta-commentary.';
          break;
        default:
          systemPrompt = 'Generate content that addresses the user request without explanations.';
      }
      
      const response = await anthropic.messages.create({
        model: 'claude-3-sonnet-20240229',
        system: systemPrompt,
        max_tokens: 2000,
        messages: [{ role: 'user', content: query }],
      });
      
      return NextResponse.json({ content: response.content[0].text });
      
    } else {
      // Default to chat response
      const response = anthropic.messages.stream({
        model: 'claude-3-sonnet-20240229',
        system: 'You are a helpful AI assistant for software engineers.',
        max_tokens: 2000,
        messages,
        tools: [sqlQueryTool, logQueryTool, metricQueryTool, s3QueryTool, dataSourceContextTool],
      });
      
      // Handle tool calls in the stream
      const responseWithTools = response.onToolCall(async (toolCall) => {
        const result = await handleToolCall(toolCall);
        return { content: JSON.stringify(result) };
      });
      
      // Return a streaming response
      return new StreamingTextResponse(responseWithTools);
    }
  } catch (error: any) {
    console.error('Error in AI route:', error);
    return NextResponse.json(
      { error: error.message || 'An error occurred' },
      { status: 500 }
    );
  }
}