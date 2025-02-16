import styled, { keyframes } from "styled-components"

const spin = keyframes`
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
`

export const LoadingSpinner = styled.div`
	width: 24px;
	height: 24px;
	border: 2px solid var(--secondary-bg);
	border-top: 2px solid var(--accent-color);
	border-radius: 50%;
	animation: ${spin} 0.8s linear infinite;
	margin: ${(props) => (props.center ? "auto" : "0")};
`
