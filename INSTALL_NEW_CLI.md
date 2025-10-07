# Install New Databricks CLI (Windows)

## Issue
You have the old Databricks CLI (v0.18.0) which doesn't support Asset Bundles (DAB). 
You need the **new unified Databricks CLI** (v0.200.0+).

## Solution: Install New Databricks CLI

### Option 1: PowerShell Installation Script (Recommended)

Open **PowerShell as Administrator** and run:

```powershell
# Download and install the latest Databricks CLI
Invoke-WebRequest -Uri "https://github.com/databricks/cli/releases/latest/download/databricks_cli_windows_amd64.zip" -OutFile "$env:TEMP\databricks_cli.zip"

# Extract to a permanent location
Expand-Archive -Path "$env:TEMP\databricks_cli.zip" -DestinationPath "$env:LOCALAPPDATA\databricks" -Force

# Add to PATH (requires admin)
$oldPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$newPath = "$oldPath;$env:LOCALAPPDATA\databricks"
[Environment]::SetEnvironmentVariable('Path', $newPath, 'User')

# Clean up
Remove-Item "$env:TEMP\databricks_cli.zip"

Write-Host "✅ Databricks CLI installed successfully!"
Write-Host "⚠️  Please restart your terminal/VS Code for PATH changes to take effect"
```

### Option 2: Manual Download

1. **Download** the latest release:
   - Go to: https://github.com/databricks/cli/releases/latest
   - Download: `databricks_cli_windows_amd64.zip`

2. **Extract** the ZIP file:
   - Extract to: `C:\Program Files\Databricks\` (or any folder)

3. **Add to PATH**:
   - Open "Environment Variables" in Windows
   - Edit "Path" under User variables
   - Add the folder where you extracted `databricks.exe`
   - Click OK

4. **Restart** your terminal/VS Code

### Option 3: Using Winget (if available)

```powershell
winget install Databricks.DatabricksCLI
```

## Verify Installation

After installation and restarting your terminal:

```bash
# Check version (should be 0.200.0+)
databricks --version

# Should show bundle command
databricks bundle --help
```

Expected output:
```
Databricks CLI v0.xxx.x
```

## Configure Authentication

The new CLI uses your existing `.databrickscfg` file:

```bash
# Test connection
databricks auth profiles

# Should show your configured profiles
```

## Next Steps

Once installed:

```bash
cd "C:\Users\victor.idakwo\Documents\ehealth Africa\ehealth Africa\eHA GitHub\Mian Disease Modelling\Malaria\Clinical_Reinforcement_learning"

# Validate bundle
databricks bundle validate --target dev

# Deploy
databricks bundle deploy --target dev
```

## Troubleshooting

### "databricks command not found"
- **Solution**: Restart your terminal/VS Code after installation
- Verify PATH was updated correctly

### "bundle command not recognized"
- **Solution**: You may still have the old CLI cached
- Run: `Get-Command databricks` to see which version is being used
- Make sure the new CLI location is first in PATH

### Still having issues?
- Close ALL terminal windows and VS Code
- Reopen VS Code
- Try the commands again

---

**After Installation**: Continue with the deployment steps from QUICKSTART.md
