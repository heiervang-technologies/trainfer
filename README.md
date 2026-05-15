# Snail Core Template

Template repository for setting up a Snail AI agent on your GitHub repository.

## Quick Start

1. **Create a new repository from this template**
   - Click the green "Use this template" button above
   - Choose a name for your repository
   - Click "Create repository"

2. **Configure your credentials**
   - After creation, a GitHub issue will automatically be created with setup instructions
   - Follow the instructions in that issue to configure the required secrets

3. **Customize your agent**
   - Edit `.github/workflows/mention-trigger.yml` to change:
     - `mention_handle` from `marksverdhai` to your agent's username
     - `agent_name` to match your agent's username
   - Edit `.github/workflows/assignment-trigger.yml` (if using assignments):
     - Update the `if` condition assignee filter to match your agent's username
     - Update `agent_name` parameter

4. **Test your agent**
   - **Via mention**: Create an issue and mention your agent (e.g., `@your-agent help me with...`)
   - **Via assignment**: Assign an issue or PR to your agent
   - The agent should respond within a few minutes

## Required Secrets

These are provided as organization secrets in heiervang-technologies and automatically inherited via `secrets: inherit`:

| Secret | Description |
|--------|-------------|
| `CLAUDE_CODE_OAUTH_TOKEN` | Claude Code OAuth token for authentication |
| `HEI_DOCKER_PAT` or `DOCKER_PAT` | Docker Hub PAT for pulling snail images |
| `HAI_GH_PAT` or `GH_PAT` | GitHub Personal Access Token with `repo` and `workflow` scopes |

For other organizations, you'll need to configure these secrets at the organization or repository level.

## Included Workflows

### Mention Trigger (`mention-trigger.yml`)

Triggers the snail agent when mentioned in:
- Issue bodies (when issue is opened)
- Issue comments
- Pull request review comments

**Key features:**
- Uses the reusable `mention-trigger-reusable.yml` workflow from `heiervang-technologies/core`
- Automatically posts progress tracking comments
- Adds reaction emojis (eyes while processing, rocket on success)

### Assignment Trigger (`assignment-trigger.yml`)

Triggers the snail agent when issues or PRs are assigned to the agent's GitHub account.

**Key features:**
- Progress tracking with visual indicators
- Automatic PR creation instructions embedded in prompt
- Support for both issues and pull requests
- PR status emoji system (🔵 in-progress, 🟢 ready, etc.)

### Setup Check (`setup-check.yml`)

Automatically runs on first push to verify:
- Required secrets are configured
- PAT has sufficient repository permissions
- Claude credentials are valid

Creates an issue with detailed setup instructions if anything is missing.

## How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Your Repository                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   @agent help me fix this bug                                        │
│         │                                                            │
│         ▼                                                            │
│   ┌─────────────────────┐                                            │
│   │ mention-trigger.yml │                                            │
│   └──────────┬──────────┘                                            │
│              │                                                       │
│              │ Uses reusable workflow                                │
│              ▼                                                       │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │           heiervang-technologies/core                        │   │
│   │         mention-trigger-reusable.yml                         │   │
│   │                     │                                        │   │
│   │                     ▼                                        │   │
│   │               spawn-agent.yml                                │   │
│   │                                                              │   │
│   │   ┌─────────────┐     ┌─────────────┐     ┌────────────┐    │   │
│   │   │ Pull snail  │────▶│ Run Claude  │────▶│ Post       │    │   │
│   │   │ container   │     │ in container│     │ results    │    │   │
│   │   └─────────────┘     └─────────────┘     └────────────┘    │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Latest Updates (January 2026)

This template has been updated to use the latest patterns from the core repository:

✅ **Mention trigger** now uses reusable workflow (`mention-trigger-reusable.yml`)
✅ **Assignment trigger** updated with latest progress tracking system
✅ **Spinner animations** in progress comments (instead of static emojis)
✅ **secrets: inherit** for cleaner secret management
✅ **Progress bar** with percentage completion
✅ **PR emoji system** for status tracking (🔵🟢🟡🔴🟣)

## Customization

### Change the Agent Username

In `.github/workflows/mention-trigger.yml`:
```yaml
with:
  mention_handle: 'your-agent-username'  # Change this
  agent_name: 'your-agent-username'       # Change this
```

In `.github/workflows/assignment-trigger.yml`:
```yaml
if: |
  github.event.assignee.login == 'your-agent-username' &&  # Change this
  (github.event.sender.login == 'your-human-username' || github.event.sender.login == 'your-agent-username')
```

And:
```yaml
with:
  agent_name: 'your-agent-username'  # Change this
```

### Use a Different Snail Image

By default, the template uses `marksverdhei/snail:builder`. To use a different image:

In `.github/workflows/mention-trigger.yml`:
```yaml
with:
  snail_image_default: 'your-registry/your-image:tag'
```

## Troubleshooting

### "Setup Required" issue keeps appearing

- Ensure all three secrets are available (org secrets or repo secrets)
- Verify the `HAI_GH_PAT` (or `GH_PAT`) has `repo` and `workflow` scopes
- Check that the PAT belongs to an account with write access to the repo

### Agent doesn't respond to mentions

1. Check the Actions tab for workflow runs
2. Look for errors in the workflow logs
3. Verify the agent username matches what's in the workflow file (`mention_handle` parameter)
4. Ensure the PAT hasn't expired

### Agent doesn't respond to assignments

1. Verify the assignee filter in `assignment-trigger.yml` matches your agent's username
2. Check that you're assigning to the correct GitHub account
3. Look at the Actions tab to see if the workflow was triggered

### Authentication errors

If you see "authentication error" in the workflow logs:
- Your Claude OAuth token may have expired
- Refresh the `CLAUDE_CODE_OAUTH_TOKEN` secret with a new token from Claude Code

### Progress tracking not updating

- Ensure the agent has the progress comment ID
- Check workflow logs for API errors
- Verify `GITHUB_TOKEN` has sufficient permissions

## Contributing

This template is maintained as part of the [heiervang-technologies/core](https://github.com/heiervang-technologies/core) project. For issues or improvements, please open an issue in the core repository.

## License

MIT
