import React, { useEffect, useRef } from "react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"

interface SentimentData {
	timestamp: number
	sentiment: number
	commentCount: number
}

interface Props {
	data: SentimentData[]
	loading?: boolean
}

export const SentimentChart: React.FC<Props> = ({ data, loading }) => {
	const chartRef = useRef(null)

	useEffect(() => {
		if (chartRef.current) {
			chartRef.current.chartInstance.update()
		}
	}, [data])

	if (loading) {
		return <VSCodeProgressRing />
	}

	return (
		<div className="chart-container">
			<ResponsiveContainer width="100%" height={300}>
				<LineChart ref={chartRef} data={data}>
					<CartesianGrid strokeDasharray="3 3" />
					<XAxis
						dataKey="timestamp"
						tickFormatter={(timestamp) => new Date(timestamp).toLocaleDateString()}
					/>
					<YAxis domain={[-1, 1]} />
					<Tooltip
						labelFormatter={(timestamp) => new Date(timestamp).toLocaleString()}
						formatter={(value) => [Number(value).toFixed(2), "Sentiment"]}
					/>
					<Line type="monotone" dataKey="sentiment" stroke="#8884d8" strokeWidth={2} dot={false} />
				</LineChart>
			</ResponsiveContainer>
		</div>
	)
}
