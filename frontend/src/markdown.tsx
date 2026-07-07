// Minimal markdown renderer for LLM narrative output — deliberately tiny and
// dependency-free (matches this project's lean-deps philosophy). Supports the
// exact subset the prompts are instructed to emit: ## / ### headings, | pipe |
// tables, - bullets, 1. numbered lists, --- rules, and **bold** inline. Not a
// general markdown engine; unknown syntax falls through as plain text.
import type { ReactNode } from 'react'

function inline(text: string, keyBase: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((p, j) =>
    p.startsWith('**') && p.endsWith('**')
      ? <b key={`${keyBase}b${j}`}>{p.slice(2, -2)}</b>
      : <span key={`${keyBase}s${j}`}>{p}</span>,
  )
}

function isTableLine(line: string): boolean {
  const t = line.trim()
  return t.startsWith('|') && t.endsWith('|') && t.length > 2
}

function isSeparatorRow(line: string): boolean {
  // | --- | :--- | ---: |
  return /^\|(\s*:?-{2,}:?\s*\|)+$/.test(line.trim())
}

function tableCells(line: string): string[] {
  return line.trim().slice(1, -1).split('|').map((c) => c.trim())
}

export function renderMarkdown(text: string): ReactNode[] {
  const lines = text.split('\n')
  const out: ReactNode[] = []
  let i = 0
  let key = 0

  while (i < lines.length) {
    const line = lines[i]
    const t = line.trim()

    if (!t) { i += 1; continue }

    if (t.startsWith('### ')) {
      out.push(<h4 key={key++} className="md-h">{inline(t.slice(4), `h${key}`)}</h4>)
      i += 1
    } else if (t.startsWith('## ')) {
      out.push(<h3 key={key++} className="md-h">{inline(t.slice(3), `h${key}`)}</h3>)
      i += 1
    } else if (/^-{3,}$/.test(t)) {
      out.push(<hr key={key++} className="md-hr" />)
      i += 1
    } else if (isTableLine(t)) {
      const rows: string[][] = []
      let header: string[] | null = null
      while (i < lines.length && isTableLine(lines[i])) {
        if (isSeparatorRow(lines[i])) {
          if (rows.length === 1 && header === null) header = rows.pop()!
        } else {
          rows.push(tableCells(lines[i]))
        }
        i += 1
      }
      out.push(
        <table key={key++} className="md-table">
          {header && (
            <thead>
              <tr>{header.map((c, j) => <th key={j}>{inline(c, `th${key}${j}`)}</th>)}</tr>
            </thead>
          )}
          <tbody>
            {rows.map((r, ri) => (
              <tr key={ri}>{r.map((c, j) => <td key={j}>{inline(c, `td${key}${ri}${j}`)}</td>)}</tr>
            ))}
          </tbody>
        </table>,
      )
    } else if (/^[-*]\s+/.test(t)) {
      const items: string[] = []
      while (i < lines.length && /^[-*]\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*]\s+/, ''))
        i += 1
      }
      out.push(
        <ul key={key++} className="md-list">
          {items.map((it, j) => <li key={j}>{inline(it, `ul${key}${j}`)}</li>)}
        </ul>,
      )
    } else if (/^\d+\.\s+/.test(t)) {
      const items: string[] = []
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ''))
        i += 1
      }
      out.push(
        <ol key={key++} className="md-list">
          {items.map((it, j) => <li key={j}>{inline(it, `ol${key}${j}`)}</li>)}
        </ol>,
      )
    } else {
      out.push(<p key={key++} className="nar-line">{inline(t, `p${key}`)}</p>)
      i += 1
    }
  }
  return out
}
