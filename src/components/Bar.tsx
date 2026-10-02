export function Bar({ value }: { value: number }) {
  const v = Math.max(0, Math.min(1, value))
  return (
    <div className="bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(v * 100)}>
      <div style={{ transform: `scaleX(${v})` }} />
    </div>
  )
}
