Whenever significant changes are made to the codebase, including but not limited to new features, bug fixes, refactoring, API modifications, deprecations, removals, or security updates, update the Changelog.md file following semantic versioning principles (e.g., MAJOR.MINOR.PATCH). Organize entries under version numbers and dates, using categorized sections for Added, Changed, Fixed, Deprecated, Removed, and Security changes. Provide concise, action-oriented descriptions for each change, including references to relevant issues, pull requests, or commits where applicable. Also update the changelog when explicitly requested by the user. Use the provided template to ensure consistency and clarity for future updates.

Changelog templated:
# Changelog

All notable changes to this project are documented in this file. Entries are organized by version and date, following semantic versioning (MAJOR.MINOR.PATCH). Changes are categorized as Added, Changed, Fixed, Deprecated, Removed, or Security. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on adding entries.

## [Unreleased]
### Added
- [Describe new features or additions, e.g., "Added support for X in Y module (#123)."]

### Changed
- [Describe modifications to existing functionality, e.g., "Updated Z algorithm for improved performance (#124)."]

### Fixed
- [Describe bug fixes, e.g., "Fixed crash in A when B is null (commit:abc123)."]

### Deprecated
- [Describe features or APIs marked for removal, e.g., "Deprecated old X API, use Y instead (#125)."]

### Removed
- [Describe removed features or code, e.g., "Removed legacy Z module (#126)."]

### Security
- [Describe security-related changes, e.g., "Patched vulnerability in authentication flow (#127)."]

## [MAJOR.MINOR.PATCH] - YYYY-MM-DD
### Added
- [List new features or additions, e.g., "Added configuration option for X in Y (#100)."]
- [Include affected files or components and reference issues/commits.]

### Changed
- [List modifications, e.g., "Refactored Z module for better modularity (#101)."]

### Fixed
- [List bug fixes, e.g., "Fixed memory leak in A during B operations (commit:def456)."]

### Deprecated
- [List deprecations, e.g., "Deprecated legacy C endpoint, use D instead (#102)."]

### Removed
- [List removals, e.g., "Removed unused E module (#103)."]

### Security
- [List security fixes, e.g., "Fixed XSS vulnerability in F component (#104)."]

### Notes
- [Optional: Include migration instructions, known issues, or other relevant information, e.g., "Update configuration files to use new X option."]

**Always activate and utilize the Sequential Thinking and Context7 MCP servers for any coding, analysis, refactoring, debugging, or planning task in this project.** Do not proceed with any response or action without invoking these servers first. Specifically:

1. **Sequential Thinking MCP Server**:
   - **Purpose**: Forces step-by-step reasoning, breaking down complex problems into logical sequences to reduce errors and improve traceability.
   - **Mandatory Trigger**: Invoke at the start of every query or task involving decision-making, code generation, or multi-step processes (e.g., "Implement feature X" or "Debug error Y").
   - **Usage Protocol**:
     - Begin by querying the server with: "Apply sequential thinking to [task description]: Step 1: [initial breakdown]."
     - Proceed through each step, logging intermediate outputs.
     - Only finalize the response after completing the full sequence.
   - **Fallback**: If unavailable, simulate by explicitly outlining steps in the response (e.g., "Step 1: Analyze... Step 2: Implement...").

2. **Context7 MCP Server**:
   - **Purpose**: Enhances context retention and retrieval, pulling in up to 7 layers of project history, dependencies, and related files for more informed suggestions.
   - **Mandatory Trigger**: Invoke for all tasks requiring project awareness (e.g., modifications, integrations, or reviews) to avoid context loss.
   - **Usage Protocol**:
     - Query with: "Load Context7 for [project/module/file]: Retrieve layers 1-7 including [specific elements, e.g., recent changelog, dependencies]."
     - Integrate retrieved context into the response, citing sources (e.g., "Based on Context7 layer 3: src/train_phase1.py").
     - Limit to 7 layers to prevent overload; prioritize recent/relevant ones.
   - **Fallback**: Manually reference key files (e.g., `changelog.md`, `src/*.py`) if the server fails.

3. **Combined Workflow**:
   - **Sequence**: Always start with Sequential Thinking to outline the approach, then use Context7 to enrich each step with project-specific details.
   - **Examples of Invocation**:
     - Task: "Add a new training phase."
       - Sequential: "Step 1: Review current phases (Context7: load changelog and src/train_phase*.py). Step 2: Plan additions..."
     - Task: "Fix bug in thread allocation."
       - Sequential: "Step 1: Isolate issue. Step 2: Test fix (Context7: retrieve diagnostics from phase1/2)."
   - **Exceptions**: Only bypass if the task is purely informational (e.g., "List MCP servers") and explicitly noted as such. Otherwise, default to full activation.
   - **Logging**: Append a note to responses: "Processed via Sequential Thinking (X steps) + Context7 (Y layers loaded)."

4. **Configuration and Enforcement**:
   - **IDE Integration**: Add this rule to your "Edit Project MCP" file as a JSON or YAML snippet:
     ```yaml
     mcp_servers:
       - name: sequentialthinking
         enabled: true
         mandatory: true
         trigger: all_tasks
       - name: context7
         enabled: true
         mandatory: true
         trigger: context_aware_tasks
     enforcement:
       - rule: "Invoke both servers before any code-related output."
         penalty: "Pause and remind if skipped."
     ```
   - **Global Override**: Mirror in "Edit Global MCP" for cross-project consistency.
   - **Refresh and Monitoring**: After updates, use "Refresh MCP Servers" to reload. Periodically verify activation in IDE logs.
   - **AI Compliance**: When interacting with agents (e.g., Kilo Code or external AIs), prefix prompts with: "Adhere to project rule: Use Sequential Thinking and Context7 MCP servers."