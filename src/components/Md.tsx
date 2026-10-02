import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeSlug from 'rehype-slug'
import 'katex/dist/katex.min.css'

const remark = [remarkGfm, remarkMath]
const rehype = [rehypeKatex]
const rehypeDoc = [rehypeSlug, rehypeKatex]

export function Md({ children, inline = false, doc = false }: { children: string; inline?: boolean; doc?: boolean }) {
  return (
    <div className={inline ? 'md md-inline' : doc ? 'md md-doc' : 'md'}>
      <ReactMarkdown
        remarkPlugins={remark}
        rehypePlugins={doc ? rehypeDoc : rehype}
        components={inline ? { p: ({ children }) => <span>{children}</span> } : undefined}
      >
        {children}
      </ReactMarkdown>
    </div>
  )
}
