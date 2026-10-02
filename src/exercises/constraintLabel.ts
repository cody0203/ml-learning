import type { ConstraintSpec, Localized } from '../content/schema'

const v = (x: number[]) => `(${x.join(', ')})`
const m = (A: number[][]) => `[${A.map((r) => r.join(' ')).join('; ')}]`

export function constraintLabel(c: ConstraintSpec): Localized {
  switch (c.kind) {
    case 'det': return { vi: `det = ${c.value}`, en: `det = ${c.value}` }
    case 'rank': return { vi: `rank = ${c.value}`, en: `rank = ${c.value}` }
    case 'singular': return { vi: 'Ma trận singular (det = 0)', en: 'Singular (det = 0)' }
    case 'nonsingular': return { vi: 'Ma trận non-singular (det ≠ 0)', en: 'Non-singular (det ≠ 0)' }
    case 'rowEchelon': return { vi: 'Ở dạng bậc thang (row echelon)', en: 'In row echelon form' }
    case 'reducedRowEchelon': return { vi: 'Ở dạng bậc thang rút gọn (RREF)', en: 'In reduced row echelon form' }
    case 'noZeroEntries': return { vi: 'Không có phần tử nào bằng 0', en: 'No zero entries' }
    case 'integerEntries': return { vi: 'Các phần tử là số nguyên', en: 'Integer entries' }
    case 'nonzero': return { vi: 'Khác ma trận/vector 0', en: 'Not all zeros' }
    case 'notDiagonal': return { vi: 'Không phải ma trận chéo', en: 'Not diagonal' }
    case 'entryRange': return { vi: `Phần tử trong [${c.min}, ${c.max}]`, en: `Entries in [${c.min}, ${c.max}]` }
    case 'dot': return { vi: `u · ${v(c.with)} = ${c.value}`, en: `u · ${v(c.with)} = ${c.value}` }
    case 'norm': return { vi: `Chuẩn L${c.p === 'inf' ? '∞' : c.p} = ${+c.value.toFixed(4)}`, en: `L${c.p === 'inf' ? '∞' : c.p}-norm = ${+c.value.toFixed(4)}` }
    case 'solves': return { vi: `Là nghiệm của A·x = ${v(c.b)} với A = ${m(c.A)}`, en: `Solves A·x = ${v(c.b)} with A = ${m(c.A)}` }
    case 'mapsTo': return { vi: `A·${v(c.x)} = ${v(c.b)}`, en: `A·${v(c.x)} = ${v(c.b)}` }
    case 'parallelTo': return { vi: `Cùng phương với ${v(c.of)}`, en: `Parallel to ${v(c.of)}` }
    case 'notParallelTo': return { vi: `Không cùng phương với ${v(c.of)}`, en: `Not parallel to ${v(c.of)}` }
    case 'noSolutionWith': return { vi: `Hệ với vế phải ${v(c.b)} vô nghiệm`, en: `System with RHS ${v(c.b)} has no solution` }
    case 'infiniteSolutionsWith': return { vi: `Hệ với vế phải ${v(c.b)} có vô số nghiệm`, en: `System with RHS ${v(c.b)} has infinitely many solutions` }
    case 'uniqueSolutionWith': return { vi: `Hệ với vế phải ${v(c.b)} có đúng 1 nghiệm`, en: `System with RHS ${v(c.b)} has exactly one solution` }
    case 'mean': return { vi: `Trung bình (mean) = ${c.value}`, en: `Mean = ${c.value}` }
    case 'median': return { vi: `Trung vị (median) = ${c.value}`, en: `Median = ${c.value}` }
    case 'variance': return { vi: `Phương sai ${c.sample ? 'mẫu (chia n−1)' : 'tổng thể (chia n)'} = ${c.value}`, en: `${c.sample ? 'Sample (÷ n−1)' : 'Population (÷ n)'} variance = ${c.value}` }
    case 'std': return { vi: `Độ lệch chuẩn ${c.sample ? 'mẫu' : 'tổng thể'} = ${+c.value.toFixed(4)}`, en: `${c.sample ? 'Sample' : 'Population'} std = ${+c.value.toFixed(4)}` }
    case 'sum': return { vi: `Tổng = ${c.value}`, en: `Sum = ${c.value}` }
    case 'range': return { vi: `Khoảng biến thiên (max − min) = ${c.value}`, en: `Range (max − min) = ${c.value}` }
    case 'meanGreaterThanMedian': return { vi: 'Mean > median (lệch phải)', en: 'Mean > median (right-skewed)' }
    case 'meanLessThanMedian': return { vi: 'Mean < median (lệch trái)', en: 'Mean < median (left-skewed)' }
    case 'nonnegative': return { vi: 'Mọi giá trị ≥ 0', en: 'All values ≥ 0' }
    case 'distinctValues': return { vi: `Có ít nhất ${c.min} giá trị khác nhau`, en: `At least ${c.min} distinct values` }
    case 'probabilityVector': return { vi: 'Là phân phối xác suất hợp lệ (≥ 0, tổng = 1)', en: 'Valid probability distribution (≥ 0, sums to 1)' }
    case 'expectation': return { vi: `Với giá trị ${v(c.values)}: E[X] = ${c.value}`, en: `For values ${v(c.values)}: E[X] = ${c.value}` }
  }
}
