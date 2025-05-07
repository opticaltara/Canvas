import nextJest from 'next/jest.js'

const createJestConfig = nextJest({
  // Provide the path to your Next.js app to load next.config.js and .env files in your test environment
  dir: './',
})

// Add any custom config to be passed to Jest
/** @type {import('jest').Config} */
const config = {
  // Add more setup options before each test is run
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  testEnvironment: 'jest-environment-jsdom',
  preset: 'ts-jest',
  transformIgnorePatterns: [
    '/node_modules/(?!(react-markdown|remark-gfm|remark-parse|remark-rehype|rehype-stringify|unified|bail|trough|vfile|vfile-message|unist-util-stringify-position|unist-util-visit|unist-util-visit-parents|unist-util-is|hast-util-whitespace|hast-util-to-string|hast-util-is-element|hastscript|web-namespaces|mdast-util-to-string|mdast-util-to-hast|mdast-util-from-markdown|micromark|micromark-util-combine-extensions|micromark-util-symbol|micromark-util-resolve-all|micromark-util-chunked|micromark-util-character|micromark-factory-whitespace|micromark-core-commonmark|decode-named-character-reference|character-entities|ccount|markdown-table|estree-util-is-identifier-name|devlop|periscopic|is-plain-obj|comma-separated-tokens|property-information)/)',
    '^.+\\.module\\.(css|sass|scss)$',
  ],
}

// createJestConfig is exported this way to ensure that next/jest can load the Next.js config which is async
export default createJestConfig(config)
