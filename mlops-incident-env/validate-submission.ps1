


#Requires -Version 5.1
<#
.SYNOPSIS
OpenEnv Submission Validator (PowerShell version)

.DESCRIPTION
Checks that your HF Space is live, Docker image builds, and openenv validate passes.

.PREREQUISITES
- Docker: https://docs.docker.com/get-docker/
- openenv-core: pip install openenv-core

.PARAMETER PingUrl
Your HuggingFace Space URL (e.g., https://your-space.hf.space)

.PARAMETER RepoDir
Path to your repo (default: current directory)

.EXAMPLE
.\validate-submission.ps1 -PingUrl 'https://my-team.hf.space'
.\validate-submission.ps1 -PingUrl 'https://my-team.hf.space' -RepoDir './my-repo'
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$PingUrl,
    
    [Parameter(Mandatory=$false)]
    [string]$RepoDir = '.'
)

$ErrorActionPreference = 'Stop'

# Color codes
$RED = "`e[0;31m"
$GREEN = "`e[0;32m"
$YELLOW = "`e[1;33m"
$BOLD = "`e[1m"
$NC = "`e[0m"

$PASS = 0

function Cleanup {
    # Cleanup logic here if needed
}

trap { Cleanup }

# Validate parameters
if ([string]::IsNullOrWhiteSpace($PingUrl)) {
    Write-Host "Usage: .\validate-submission.ps1 -PingUrl <ping_url> [-RepoDir <repo_dir>]"
    Write-Host ""
    Write-Host "  PingUrl   Your HuggingFace Space URL (e.g., https://your-space.hf.space)"
    Write-Host "  RepoDir   Path to your repo (default: current directory)"
    exit 1
}

try {
    $RepoDir = (Resolve-Path $RepoDir).Path
} catch {
    Write-Host "Error: directory '$RepoDir' not found"
    exit 1
}

$PingUrl = $PingUrl.TrimEnd('/')
$env:PING_URL = $PingUrl

function Log {
    param([string]$Message)
    $time = Get-Date -Format "HH:mm:ss"
    Write-Host "[$time] $Message" -NoNewline
}

function Pass {
    param([string]$Message)
    Log "${GREEN}PASSED${NC} -- $Message"
    Write-Host ""
    $script:PASS++
}

function Fail {
    param([string]$Message)
    Log "${RED}FAILED${NC} -- $Message"
    Write-Host ""
}

function Hint {
    param([string]$Message)
    Write-Host "  ${YELLOW}Hint:${NC} $Message"
}

function StopAt {
    param([string]$Stage)
    Write-Host ""
    Write-Host "${RED}${BOLD}Validation stopped at $Stage.${NC} Fix the above before continuing."
    Cleanup
    exit 1
}

# Main validation
Write-Host ""
Write-Host "${BOLD}========================================${NC}"
Write-Host "${BOLD}  OpenEnv Submission Validator${NC}"
Write-Host "${BOLD}========================================${NC}"
Log "Repo:     $RepoDir"
Write-Host ""
Log "Ping URL: $PingUrl"
Write-Host ""
Write-Host ""

# Step 1: Ping HF Space
Log "${BOLD}Step 1/3: Pinging HF Space${NC} ($PingUrl/reset) ..."
Write-Host ""

try {
    $response = Invoke-WebRequest -Uri "$PingUrl/reset" `
        -Method POST `
        -Headers @{'Content-Type' = 'application/json'} `
        -Body '{}' `
        -TimeoutSec 30 `
        -UseBasicParsing `
        -ErrorAction Stop
    
    $HTTP_CODE = $response.StatusCode
} catch {
    $HTTP_CODE = $_.Exception.Response.StatusCode.Value
    if ($null -eq $HTTP_CODE) {
        $HTTP_CODE = 0
    }
}

if ($HTTP_CODE -eq 200) {
    Pass "HF Space is live and responds to /reset"
} elseif ($HTTP_CODE -eq 0) {
    Fail "HF Space not reachable (connection failed or timed out)"
    Hint "Check your network connection and that the Space is running."
    Hint "Try: curl -s -o `$null -w '%{http_code}' -X POST $PingUrl/reset"
    StopAt "Step 1"
} else {
    Fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
    Hint "Make sure your Space is running and the URL is correct."
    Hint "Try opening $PingUrl in your browser first."
    StopAt "Step 1"
}

# Step 2: Check Docker and build
Log "${BOLD}Step 2/3: Running docker build${NC} ..."
Write-Host ""

try {
    $null = Get-Command docker -ErrorAction Stop
} catch {
    Fail "docker command not found"
    Hint "Install Docker: https://docs.docker.com/get-docker/"
    StopAt "Step 2"
}

if (Test-Path "$RepoDir/Dockerfile") {
    $DOCKER_CONTEXT = $RepoDir
} elseif (Test-Path "$RepoDir/server/Dockerfile") {
    $DOCKER_CONTEXT = "$RepoDir/server"
} else {
    Fail "No Dockerfile found in repo root or server/ directory"
    StopAt "Step 2"
}

Log "  Found Dockerfile in $DOCKER_CONTEXT"
Write-Host ""

try {
    $BUILD_OUTPUT = & docker build $DOCKER_CONTEXT 2>&1
    
    if ($LASTEXITCODE -ne 0) {
        Fail "Docker build failed"
        $BUILD_OUTPUT | Select-Object -Last 20 | ForEach-Object { Write-Host $_ }
        StopAt "Step 2"
    }
    
    Pass "Docker build succeeded"
} catch {
    Fail "Docker build failed: $_"
    StopAt "Step 2"
}

# Step 3: Run openenv validate
Log "${BOLD}Step 3/3: Running openenv validate${NC} ..."
Write-Host ""

try {
    $null = Get-Command openenv -ErrorAction Stop
} catch {
    Fail "openenv command not found"
    Hint "Install it: pip install openenv-core"
    StopAt "Step 3"
}

try {
    Push-Location $RepoDir
    $VALIDATE_OUTPUT = & openenv validate 2>&1
    Pop-Location
    
    if ($LASTEXITCODE -ne 0) {
        Fail "openenv validate failed"
        $VALIDATE_OUTPUT | ForEach-Object { Write-Host $_ }
        StopAt "Step 3"
    }
    
    Pass "openenv validate passed"
    if ($VALIDATE_OUTPUT) {
        Log "  $VALIDATE_OUTPUT"
        Write-Host ""
    }
} catch {
    Fail "openenv validate failed: $_"
    StopAt "Step 3"
}

# Success
Write-Host ""
Write-Host "${BOLD}========================================${NC}"
Write-Host "${GREEN}${BOLD}  All 3/3 checks passed!${NC}"
Write-Host "${GREEN}${BOLD}  Your submission is ready to submit.${NC}"
Write-Host "${BOLD}========================================${NC}"
Write-Host ""

Cleanup
exit 0