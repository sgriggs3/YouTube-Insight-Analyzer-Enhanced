import styled from "styled-components"

export const Card = styled.div<{ interactive?: boolean }>`
	background: var(--secondary-bg);
	border-radius: 12px;
	padding: 16px;
	margin: 8px 0;
	transition:
		transform 0.2s ease,
		box-shadow 0.2s ease;

	${(props) =>
		props.interactive &&
		`
    cursor: pointer;
    &:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
  `}

	@media (max-width: 768px) {
		padding: 12px;
	}
`
