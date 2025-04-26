import type React from "react"
import { Bot } from "lucide-react"

const EmptyCanvas: React.FC = () => {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center p-4">
      <Bot className="w-16 h-16 text-gray-400 mb-4" />
      <h2 className="text-2xl font-bold text-gray-700 mb-2">This canvas is empty</h2>
      <p className="text-gray-500 mb-4">Start your analysis by asking Sherlog a question</p>
    </div>
  )
}

export default EmptyCanvas
