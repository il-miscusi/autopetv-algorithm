#!/bin/bash

# Script to check if nnUNet weights are properly downloaded
# If not (e.g., git-lfs not run), downloads from Google Drive

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
CHECKPOINT_PATH="$SCRIPTPATH/nnUNet_results/Dataset998_AutoPETV/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
WEIGHTS_ZIP="$SCRIPTPATH/weights.zip"
GDRIVE_ID="1G0HGHzQMXzslGDxFSNs5fq3RCeAu7M6l"
GDRIVE_URL="https://drive.google.com/uc?export=download&id=${GDRIVE_ID}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "==============================================="
echo "Checking nnUNet model weights..."
echo "==============================================="

# Function to check if file is a git-lfs pointer
is_git_lfs_pointer() {
    local file="$1"
    
    # Check if file exists
    if [ ! -f "$file" ]; then
        return 1  # File doesn't exist
    fi
    
    # Check file size (git-lfs pointers are typically < 200 bytes)
    local filesize=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
    if [ "$filesize" -lt 1000 ]; then
        # Check if it starts with git-lfs marker
        if head -n 1 "$file" 2>/dev/null | grep -q "version https://git-lfs.github.com"; then
            return 0  # Is a git-lfs pointer
        fi
    fi
    
    return 1  # Not a git-lfs pointer
}

# Function to download from Google Drive
download_from_gdrive() {
    echo -e "${YELLOW}Downloading weights from Google Drive...${NC}"
    
    # Try using gdown if available
    if command -v gdown &> /dev/null; then
        echo "Using gdown for download..."
        gdown "$GDRIVE_ID" -O "$WEIGHTS_ZIP"
        return $?
    fi
    
    # Fallback to wget
    if command -v wget &> /dev/null; then
        echo "Using wget for download..."
        wget --no-check-certificate "$GDRIVE_URL" -O "$WEIGHTS_ZIP"
        return $?
    fi
    
    # Fallback to curl
    if command -v curl &> /dev/null; then
        echo "Using curl for download..."
        curl -L "$GDRIVE_URL" -o "$WEIGHTS_ZIP"
        return $?
    fi
    
    echo -e "${RED}ERROR: No download tool available (gdown, wget, or curl)${NC}"
    echo "Please install one of: gdown (pip install gdown), wget, or curl"
    return 1
}

# Function to extract weights
extract_weights() {
    echo -e "${YELLOW}Extracting weights...${NC}"
    
    if [ ! -f "$WEIGHTS_ZIP" ]; then
        echo -e "${RED}ERROR: weights.zip not found at $WEIGHTS_ZIP${NC}"
        return 1
    fi
    
    # Remove old nnUNet_results if it exists
    if [ -d "$SCRIPTPATH/nnUNet_results" ]; then
        echo "Removing old nnUNet_results directory..."
        rm -rf "$SCRIPTPATH/nnUNet_results"
    fi
    
    # Extract
    cd "$SCRIPTPATH"
    unzip -q "$WEIGHTS_ZIP"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Extraction successful!${NC}"
        return 0
    else
        echo -e "${RED}ERROR: Failed to extract weights.zip${NC}"
        return 1
    fi
}

# Main logic
NEEDS_DOWNLOAD=false

# Check if checkpoint file exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo -e "${YELLOW}Checkpoint file not found: $CHECKPOINT_PATH${NC}"
    NEEDS_DOWNLOAD=true
elif is_git_lfs_pointer "$CHECKPOINT_PATH"; then
    echo -e "${YELLOW}Checkpoint file is a git-lfs pointer (not downloaded)${NC}"
    echo "This usually means 'git lfs pull' was not run or git-lfs limit was reached."
    NEEDS_DOWNLOAD=true
else
    # Check file size to ensure it's a valid checkpoint (should be > 100MB)
    filesize=$(stat -f%z "$CHECKPOINT_PATH" 2>/dev/null || stat -c%s "$CHECKPOINT_PATH" 2>/dev/null)
    filesize_mb=$((filesize / 1024 / 1024))
    
    if [ "$filesize_mb" -lt 100 ]; then
        echo -e "${YELLOW}Checkpoint file seems too small (${filesize_mb}MB). Expected > 100MB.${NC}"
        NEEDS_DOWNLOAD=true
    else
        echo -e "${GREEN}✓ Checkpoint file exists and appears valid (${filesize_mb}MB)${NC}"
    fi
fi

# Download and extract if needed
if [ "$NEEDS_DOWNLOAD" = true ]; then
    echo ""
    echo -e "${YELLOW}Attempting to download and extract weights...${NC}"
    
    # Check if weights.zip already exists and is valid
    if [ -f "$WEIGHTS_ZIP" ]; then
        zip_size=$(stat -f%z "$WEIGHTS_ZIP" 2>/dev/null || stat -c%s "$WEIGHTS_ZIP" 2>/dev/null)
        zip_size_mb=$((zip_size / 1024 / 1024))
        
        if [ "$zip_size_mb" -gt 100 ]; then
            echo -e "${GREEN}Found existing weights.zip (${zip_size_mb}MB), skipping download${NC}"
        else
            echo -e "${YELLOW}Existing weights.zip seems invalid (${zip_size_mb}MB), re-downloading...${NC}"
            rm -f "$WEIGHTS_ZIP"
            download_from_gdrive
            if [ $? -ne 0 ]; then
                echo -e "${RED}Failed to download weights${NC}"
                exit 1
            fi
        fi
    else
        download_from_gdrive
        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to download weights${NC}"
            exit 1
        fi
    fi
    
    # Extract
    extract_weights
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to extract weights${NC}"
        exit 1
    fi
    
    # Verify extraction
    if [ -f "$CHECKPOINT_PATH" ]; then
        filesize=$(stat -f%z "$CHECKPOINT_PATH" 2>/dev/null || stat -c%s "$CHECKPOINT_PATH" 2>/dev/null)
        filesize_mb=$((filesize / 1024 / 1024))
        echo -e "${GREEN}✓ Weights successfully downloaded and extracted!${NC}"
        echo -e "${GREEN}✓ Checkpoint file size: ${filesize_mb}MB${NC}"
    else
        echo -e "${RED}ERROR: Checkpoint file still not found after extraction${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}Weight check complete! Ready to proceed.${NC}"
echo -e "${GREEN}===============================================${NC}"
exit 0
