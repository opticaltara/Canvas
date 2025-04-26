"use client"

import type React from "react"

import { Card, Title, AreaChart, BarChart, LineChart, DonutChart } from "@tremor/react"

interface ChartData {
  [key: string]: string | number
}

interface TremorChartProps {
  title: string
  data: ChartData[]
  type: "area" | "bar" | "line" | "donut"
  categories: string[]
  index: string
  colors?: string[]
  valueFormatter?: (number: number) => string
  showLegend?: boolean
  showAnimation?: boolean
  yAxisWidth?: number
}

const TremorChart: React.FC<TremorChartProps> = ({
  title,
  data,
  type,
  categories,
  index,
  colors = ["blue", "green", "red", "yellow", "purple"],
  valueFormatter,
  showLegend = true,
  showAnimation = true,
  yAxisWidth = 40,
}) => {
  const renderChart = () => {
    const commonProps = {
      data,
      categories,
      index,
      colors,
      valueFormatter,
      showLegend,
      showAnimation,
      yAxisWidth,
    }

    switch (type) {
      case "area":
        return <AreaChart {...commonProps} />
      case "bar":
        return <BarChart {...commonProps} />
      case "line":
        return <LineChart {...commonProps} />
      case "donut":
        return (
          <DonutChart
            data={data}
            category={categories[0]}
            index={index}
            colors={colors}
            valueFormatter={valueFormatter}
            showAnimation={showAnimation}
          />
        )
      default:
        return <LineChart {...commonProps} />
    }
  }

  return (
    <Card>
      <Title>{title}</Title>
      {renderChart()}
    </Card>
  )
}

export default TremorChart
