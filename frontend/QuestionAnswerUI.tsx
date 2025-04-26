"use client"

import type React from "react"
import { useState, useEffect, useRef } from "react"
import { User, Bot } from "lucide-react"

interface QuestionAnswerUIProps {
  question: string
  answer: string
  isStreaming: boolean
}

const QuestionAnswerUI: React.FC<QuestionAnswerUIProps> = ({ question, answer, isStreaming }) => {
  const [streamedAnswer, setStreamedAnswer] = useState("")
  const answerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isStreaming) {
      let index = 0
      const interval = setInterval(() => {
        if (index < answer.length) {
          setStreamedAnswer((prev) => prev + answer[index])
          index++
        } else {
          clearInterval(interval)
        }
      }, 50)

      return () => clearInterval(interval)
    } else {
      setStreamedAnswer(answer || "")
    }
  }, [answer, isStreaming])

  useEffect(() => {
    if (answerRef.current) {
      answerRef.current.scrollIntoView({ behavior: "smooth", block: "end" })
    }
  }, [streamedAnswer])

  return (
    <div className="mb-4 bg-gray-50 p-4 rounded-lg">
      <div className="flex items-start mb-2">
        <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center mr-2">
          <User className="w-5 h-5 text-gray-600" />
        </div>
        <div className="bg-blue-100 p-2 rounded-lg shadow flex-1">
          <p className="text-xs font-semibold mb-1 text-blue-600">User Query:</p>
          <p className="text-sm text-blue-800">{question}</p>
        </div>
      </div>
      <div className="flex items-start">
        <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center mr-2">
          <Bot className="w-5 h-5 text-gray-600" />
        </div>
        <div className="bg-green-100 p-2 rounded-lg shadow flex-1">
          <p className="text-xs font-semibold mb-1 text-green-600">Sherlog Answer:</p>
          <p className="text-sm text-green-800" ref={answerRef}>
            {streamedAnswer}
          </p>
        </div>
      </div>
    </div>
  )
}

export default QuestionAnswerUI
