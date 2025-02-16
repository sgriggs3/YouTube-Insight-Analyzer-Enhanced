import styled, { keyframes } from "styled-components"

const slideIn = keyframes`
  from { transform: translateY(100%); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
`

export const Toast = styled.div<{ type?: "success" | "error" | "info" }>`
	position: fixed;
	bottom: 20px;
	right: 20px;
	padding: 12px 24px;
	border-radius: 8px;
	background: ${(props) => {
		switch (props.type) {
			case "success":
				return "var(--success-color)"
			case "error":
				return "var(--error-color)"
			default:
				return "var(--secondary-bg)"
		}
	}};
	color: var(--text-primary);
	animation: ${slideIn} 0.3s ease-out;
	box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
	z-index: 1000;
`
