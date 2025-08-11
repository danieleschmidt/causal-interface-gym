import React from 'react'
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  IconButton,
  Menu,
  MenuItem,
} from '@mui/material'
import { Link, useLocation } from 'react-router-dom'
import { Psychology, MoreVert } from '@mui/icons-material'

export const Navbar: React.FC = () => {
  const location = useLocation()
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null)

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget)
  }

  const handleMenuClose = () => {
    setAnchorEl(null)
  }

  const isActive = (path: string) => location.pathname === path

  return (
    <AppBar position="sticky" elevation={0} sx={{ borderBottom: 1, borderColor: 'divider' }}>
      <Toolbar>
        <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
          <Psychology sx={{ mr: 2, color: 'primary.main' }} />
          <Typography
            variant="h6"
            component={Link}
            to="/"
            sx={{
              textDecoration: 'none',
              color: 'inherit',
              fontWeight: 700,
            }}
          >
            Causal Interface Gym
          </Typography>
        </Box>

        {/* Desktop Navigation */}
        <Box sx={{ display: { xs: 'none', md: 'flex' }, gap: 1 }}>
          <Button
            component={Link}
            to="/"
            color={isActive('/') ? 'primary' : 'inherit'}
            variant={isActive('/') ? 'outlined' : 'text'}
          >
            Home
          </Button>
          <Button
            component={Link}
            to="/experiment"
            color={isActive('/experiment') ? 'primary' : 'inherit'}
            variant={isActive('/experiment') ? 'outlined' : 'text'}
          >
            Experiment
          </Button>
          <Button
            component={Link}
            to="/benchmark"
            color={isActive('/benchmark') ? 'primary' : 'inherit'}
            variant={isActive('/benchmark') ? 'outlined' : 'text'}
          >
            Benchmark
          </Button>
          <Button
            component={Link}
            to="/docs"
            color={isActive('/docs') ? 'primary' : 'inherit'}
            variant={isActive('/docs') ? 'outlined' : 'text'}
          >
            Docs
          </Button>
        </Box>

        {/* Mobile Navigation */}
        <Box sx={{ display: { xs: 'flex', md: 'none' } }}>
          <IconButton
            size="large"
            aria-label="menu"
            aria-controls="menu-appbar"
            aria-haspopup="true"
            onClick={handleMenuOpen}
            color="inherit"
          >
            <MoreVert />
          </IconButton>
          <Menu
            id="menu-appbar"
            anchorEl={anchorEl}
            anchorOrigin={{
              vertical: 'bottom',
              horizontal: 'right',
            }}
            keepMounted
            transformOrigin={{
              vertical: 'top',
              horizontal: 'right',
            }}
            open={Boolean(anchorEl)}
            onClose={handleMenuClose}
            sx={{
              display: { xs: 'block', md: 'none' },
            }}
          >
            <MenuItem component={Link} to="/" onClick={handleMenuClose}>
              Home
            </MenuItem>
            <MenuItem component={Link} to="/experiment" onClick={handleMenuClose}>
              Experiment
            </MenuItem>
            <MenuItem component={Link} to="/benchmark" onClick={handleMenuClose}>
              Benchmark
            </MenuItem>
            <MenuItem component={Link} to="/docs" onClick={handleMenuClose}>
              Docs
            </MenuItem>
          </Menu>
        </Box>
      </Toolbar>
    </AppBar>
  )
}