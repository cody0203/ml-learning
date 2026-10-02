import { describe, expect, it } from 'vitest'
import { det, rank, rref, isRowEchelon, isReducedRowEchelon, classifySystem, norm } from './linalg'
import { checkConstraint } from './checkers'
import { evaluate } from './expr'
import { generators, instantiate } from './generators'
import { mulberry32 } from './random'
import { review, newSrs } from './srs'

describe('linalg', () => {
  it('det', () => {
    expect(det([[1, 2], [3, 4]])).toBeCloseTo(-2)
    expect(det([[1, 2], [2, 4]])).toBe(0)
    expect(det([[2, 0, 1], [1, 3, 2], [1, 1, 2]])).toBeCloseTo(6)
  })
  it('rank / rref', () => {
    expect(rank([[1, 2], [2, 4]])).toBe(1)
    expect(rank([[1, 0, 0], [0, 1, 0], [0, 0, 0]])).toBe(2)
    expect(rref([[2, 4], [1, 3]]).R).toEqual([[1, 0], [0, 1]])
  })
  it('echelon forms', () => {
    expect(isRowEchelon([[2, 1], [0, 3]])).toBe(true)
    expect(isRowEchelon([[0, 1], [1, 0]])).toBe(false)
    expect(isRowEchelon([[0, 0], [1, 0]])).toBe(false)
    expect(isReducedRowEchelon([[1, 0], [0, 1]])).toBe(true)
    expect(isReducedRowEchelon([[1, 2], [0, 1]])).toBe(false)
  })
  it('classify', () => {
    expect(classifySystem([[1, 1], [1, 2]], [10, 12])).toBe('unique')
    expect(classifySystem([[1, 1], [2, 2]], [10, 20])).toBe('infinite')
    expect(classifySystem([[1, 1], [2, 2]], [10, 24])).toBe('none')
  })
  it('norms', () => {
    expect(norm([3, 4])).toBe(5)
    expect(norm([3, -4], 1)).toBe(7)
  })
})

describe('checkers', () => {
  it('construct constraints', () => {
    expect(checkConstraint({ kind: 'singular' }, [[1, 2], [2, 4]]).ok).toBe(true)
    expect(checkConstraint({ kind: 'dot', with: [2, 1], value: 0 }, [[1], [-2]]).ok).toBe(true)
    expect(checkConstraint({ kind: 'dot', with: [2, 1], value: 0 }, [[1, -2]]).ok).toBe(true)
    expect(checkConstraint({ kind: 'mapsTo', x: [1, 1], b: [3, 7] }, [[1, 2], [3, 4]]).ok).toBe(true)
    expect(checkConstraint({ kind: 'noSolutionWith', b: [1, 3] }, [[1, 1], [2, 2]]).ok).toBe(true)
    expect(checkConstraint({ kind: 'det', value: 5 }, [[1, 2, 3]]).ok).toBe(false)
  })
  it('stats constraints', () => {
    expect(checkConstraint({ kind: 'mean', value: 3 }, [[1, 2, 6]]).ok).toBe(true)
    expect(checkConstraint({ kind: 'median', value: 2.5 }, [[4, 1, 2, 3]]).ok).toBe(true)
    expect(checkConstraint({ kind: 'variance', value: 1, sample: true }, [[1, 2, 3]]).ok).toBe(true)
    expect(checkConstraint({ kind: 'variance', value: 2 / 3, sample: false }, [[1, 2, 3]]).ok).toBe(true)
    expect(checkConstraint({ kind: 'meanGreaterThanMedian' }, [[1, 1, 10]]).ok).toBe(true)
    expect(checkConstraint({ kind: 'probabilityVector' }, [[0.2, 0.3, 0.5]]).ok).toBe(true)
    expect(checkConstraint({ kind: 'probabilityVector' }, [[0.6, 0.6, -0.2]]).ok).toBe(false)
    expect(checkConstraint({ kind: 'expectation', values: [0, 1, 2], value: 1.3 }, [[0.2, 0.3, 0.5]]).ok).toBe(true)
  })
})

describe('expr', () => {
  it('evaluates', () => {
    expect(evaluate('7/3')).toBeCloseTo(7 / 3)
    expect(evaluate('-2^2')).toBe(-4)
    expect(evaluate('sqrt(2)*sqrt(2)')).toBeCloseTo(2)
    expect(evaluate('2pi'.replace('2pi', '2*pi'))).toBeCloseTo(2 * Math.PI)
    expect(evaluate('1,5')).toBe(1.5)
    expect(() => evaluate('alert(1)')).toThrow()
    expect(evaluate('x^2 - 2*x', { x: 3 })).toBe(3)
    expect(evaluate('log(e)')).toBeCloseTo(1)
    expect(evaluate('sigmoid(0)')).toBe(0.5)
    expect(evaluate('log2(8)')).toBe(3)
  })
})

describe('generators', () => {
  it('all generators produce valid exercises', () => {
    for (const name of Object.keys(generators))
      for (let s = 0; s < 50; s++) {
        const ex = instantiate({ id: 'x', topic: 't', type: 'parametric', generator: name, params: {}, difficulty: 1, tags: [], hints: [], prompt: { vi: 'a', en: 'a' }, explanation: { vi: 'a', en: 'a' } }, mulberry32(s))
        expect(ex.prompt.en.length).toBeGreaterThan(5)
      }
  })
})

describe('srs', () => {
  it('schedules', () => {
    const s1 = review(newSrs(0), 'good', 0)
    expect(s1.interval).toBe(1)
    const s2 = review(s1, 'good', 0)
    expect(s2.interval).toBe(6)
    expect(review(s2, 'again', 0).reps).toBe(0)
  })
})
