var plugin = require('./index');
var base = require('@jupyter-widgets/base');

module.exports = {
  id: 'astrolab',
  requires: [base.IJupyterWidgetRegistry],
  activate: function(app, widgets) {
      widgets.registerWidget({
          name: 'astrolab',
          version: plugin.version,
          exports: plugin
      });
  },
  autoStart: true
};

