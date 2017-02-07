'use strict'

// index.js (main process)
// - GUI (renderer process)
// - GUI (renderer process)
// - GUI (renderer process)

const electron = require('electron')
const app = electron.app
const BrowserWindow = electron.BrowserWindow
const Menu = electron.Menu
const dialog = electron.dialog

let mainWindow
let dbWindow

// menu bar
let menuTemplate = [{
  label: 'MyApp',
  submenu: [
    { label: 'About', accelerator: 'CmdOrCtrl+Shift+A', click: function() { showAboutDialog(); } },
    { type: 'separator' },
    { label: 'MongoDB', accelerator: 'CmdOrCtrl+,', click: function() { showDBWindow(); }  },
    { type: 'separator' },
    { label: 'Quit', accelerator: 'CmdOrCtrl+Q', click: function() { app.quit(); }  }
  ]
}]

let menu = Menu.buildFromTemplate(menuTemplate);

function showAboutDialog() {
  dialog.showMessageBox({
    type: 'info',
    buttons: ['OK'],
    message: 'About This App',
    detail: 'This app was created by @dotinstall'
  })
}

function showDBWindow() {
  settingsWindow = new BrowserWindow({width: 1100, height: 600})
  settingsWindow.loadURL(`file://${__dirname}/db.html`)
  settingsWindow.webContents.openDevTools()
  settingsWindow.show()
  settingsWindow.on('closed', function() {
    settingsWindow = null
  })
}

function createMainWindow() {
  Menu.setApplicationMenu(menu)
  mainWindow = new BrowserWindow({width: 1100, height: 740})
  mainWindow.loadURL(`file://${__dirname}/index.html`)
  // mainWindow.webContents.openDevTools()
  mainWindow.on('closed', function() {
    mainWindow = null
  })
}

app.on('ready', function() {
  // create window
  createMainWindow()
})

app.on('window-all-closed', function() {
  // close windows
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('activate', function() {
  if (mainWindow === null) {
    createMainWindow()
  }
})
